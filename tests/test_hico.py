#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the HICO L1B reader (ISS).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

Documented deviations from the v1 implementation (see REFACTORING.md):
per-file south-up / west-to-east flip (the sample is a north-up descending
granule, so v2 row i carries raw row ny-1-i), `resolution` from the product's
own global attribute (100) instead of v1's hardcoded 30, and the added
shortname/crs attributes. Chunking matches legacy (one all-bands block).

The azimuths keep the product's absolute [0, 360) convention while the naming
registry declares [0, 180], so `generic.Test.angles` does not apply; a
scene-aware check replaces it below.
"""

import gc
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from core.geo.naming import names

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.flags import FlagsInit, GenericFlags
from eoread.reader import UnsupportedProductType
from eoread.readers.hico import HICO_BANDS, Level1_HICO

# The suite is a single process and must stay on one libhdf5 copy: opening
# the HICO file through netCDF4 here would poison the h5netcdf dimension-scale
# operations of every other reader test running later in the session
# (REFACTORING.md, dual-libhdf5 decision). The production default is
# 'netcdf4' (see Level1B_HICO.__init__); the suite pins 'h5netcdf'.
_ENGINE = 'h5netcdf'


def _check_angles(ds):
    """Sensor-native angle range check on a decimated sample (never computes
    the full grid).

    Deliberately not `generic.Test.angles`, which checks against the naming
    registry minv/maxv: HICO stores SAA/VAA as absolute degrees in [0, 360)
    while the registry declares [0, 180] (documented registry inconsistency,
    see MERIS/OLCI). The zenith angles are ordinary [0, 90] values.
    """
    sub = {str(a): ds[str(a)].isel({str(names.rows): slice(None, None, 50),
                                    str(names.columns): slice(None, None, 50)})
           for a in (names.sza, names.vza, names.saa, names.vaa)}
    limits = {names.sza: (0, 90), names.vza: (0, 90),
              names.saa: (0, 360), names.vaa: (0, 360)}
    for a, (lo, hi) in limits.items():
        v = sub[str(a)].values
        assert v.min() >= lo - 1e-3 and v.max() <= hi + 1e-3, \
            f'{a} out of range [{lo}, {hi}], got [{v.min()}, {v.max()}]'


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so reader datasets are released promptly.
    yield
    gc.collect()


@pytest.fixture(scope='session')
def hico_dir():
    return get_sample(Level1B_HICO)


@pytest.fixture(scope='session')
def hico(hico_dir):
    return Level1B_HICO(hico_dir, chunks=500, engine=_ENGINE)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------

def test_l1_instantiation(hico_dir, chunks):
    Level1B_HICO(hico_dir, chunks=chunks, engine=_ENGINE)


def test_l1_main(hico):
    generic.Test.main(hico, angle_data=True)


def test_l1_conformance(hico):
    generic.Test.v2_conformance(
        hico, primary=str(names.ltoa),
        required_vars=(str(names.sza), str(names.saa), str(names.vza),
                       str(names.vaa), 'altitude'))


def test_l1_latlon(hico):
    # Orientation check only: the swath is a tilted parallelogram (latitude
    # varies along raster rows too), so the flat-projection invariance check
    # does not apply.
    generic.Test.latlon(hico, flat_proj=False)


def test_l1_angles(hico):
    _check_angles(hico)


def test_l1_lazy_load(hico):
    generic.Test.lazy_load(hico)


def test_l1_flagreader(hico):
    generic.Test.flagreader(hico)
    # No usable QC bits in this product generation (scan_quality_flags is
    # entirely fill): every flag decodes to False.
    result = FlagsInit({GenericFlags.L1_INVALID: 4}, 'uint8',
                       hico.attrs['_flag_reader']).map_blocks(hico)
    assert (result['flags'].compute() == 0).all()


def test_l1_subset(hico_dir):
    ds = Level1B_HICO(hico_dir, chunks=500, engine=_ENGINE)
    generic.Test.subset(ds)


# --------------------------------------------------------------------------
# Payload / bands
# --------------------------------------------------------------------------

def test_l1_bands(hico):
    # 128 bands, identified by their nominal central wavelengths (as strings)
    assert len(hico.bands) == len(HICO_BANDS) == 128
    assert list(hico.bands.values) == [str(b) for b in HICO_BANDS]

    # Central wavelengths from the product, in nm
    cwav = hico['cwav'].values
    assert cwav.dtype.kind == 'f'
    assert (cwav > 300).all() and (cwav < 1100).all()
    # First and last band central wavelengths, in nm
    assert abs(cwav[0] - 352.5) < 0.5
    assert abs(cwav[-1] - 1080.) < 0.5


def test_l1_units(hico):
    assert hico[str(names.ltoa)].attrs['units'] == 'W/m^2/micrometer/sr'


def test_l1_raw_parity(hico, hico_dir):
    """Ltoa reproduces raw uint16 * scale_factor at the mirrored rows.

    The sample is north-up (raw row 0 = northern edge), so the per-file flip
    makes v2 row i carry raw row ny-1-i — pinned here against the undecoded
    file values.
    """
    tree = xr.open_datatree(hico_dir, engine=_ENGINE, decode_cf=False)
    lt = tree['products']['Lt']
    scale = np.float32(lt.attrs['scale_factor'])
    raw = lt[:, :, 8].values
    ny = hico.sizes[str(names.rows)]
    y = slice(100, 130)
    x = slice(300, 340)
    got = hico[str(names.ltoa)].isel(bands=8, y=y, x=x).compute().values
    want = (raw[ny - 1 - np.arange(y.start, y.stop)]
            .astype('float32') * scale)[:, x.start:x.stop]
    assert got.dtype == np.dtype('float32')
    np.testing.assert_array_equal(got, want)


def test_l1_altitude(hico):
    # No terrain information in the product: zero field (v1 / polymer v4 parity)
    alt = hico['altitude']
    assert alt.dims == (str(names.rows), str(names.columns))
    assert alt.dtype == np.dtype('float32')
    assert (alt.compute() == 0).all()


# --------------------------------------------------------------------------
# Attributes
# --------------------------------------------------------------------------

def test_l1_attributes(hico, hico_dir):
    a = hico.attrs
    assert a[str(names.sensor)] == 'HICO'
    assert a[str(names.platform)] == 'ISS'
    # From the product's global attribute 'resolution' = '100m'; v1 hardcoded 30
    assert a[str(names.resolution)] == 100
    assert a[str(names.product_name)] == Path(hico_dir).name
    assert a[str(names.input_directory)] == str(Path(hico_dir).parent)
    assert a[str(names.datetime)] == '2014-03-10T11:52:27'
    assert a[str(names.shortname)] == 'L1B_ISS'
    assert a[str(names.crs)] == 'EPSG:4326'
    assert a['_srf_getter'] == 'eotools.srf.get_SRF_NASA'
    assert a['_srf_getter_arg'] == 'iss_hico'
    assert a['_flag_reader'] == 'eoread.readers.hico.Level1B_HICO'


def test_l1_v1_compat(hico):
    old = v1(hico)
    # identity + schema_version strip (no HICO renames or clean breaks)
    assert str(names.ltoa) in old
    assert 'schema_version' not in old.attrs


# --------------------------------------------------------------------------
# Variant detection
# --------------------------------------------------------------------------

def test_l1_rejects_non_l1b(hico_dir, tmp_path):
    # A root-level processing_level other than L1B must be rejected loudly.
    bad = xr.Dataset(attrs={'processing_level': 'L1_ISS', 'instrument': 'HICO'})
    target = tmp_path / 'H2014069115230.L1_ISS.nc'
    bad.to_netcdf(target, engine=_ENGINE)
    with pytest.raises(UnsupportedProductType):
        Level1B_HICO(target, engine=_ENGINE)
