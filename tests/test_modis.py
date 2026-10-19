#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the MODIS L1B reader (Phase 4).

Sample (see tests/README.md):
- LEVEL1_MODISA_MYD: Aqua MYD021KM HDF4 scene, 2030 x 1354 pixels at ~1 km
  registration, 38 file-native bands ('1.0' ... '36.0', with the high-gain TDI
  pairs '13.5'/'14.5'), latitude 44.9-67.2 N.

The legacy v1 reader is kept as the reference implementation
(eoread/legacy/modis.py); these tests pin the v2 contract instead: schema
conformance, canonical south-up orientation, lazy loading and flag decoding.
MODIS has a documented clean break in eoread.compat (CLEAN_BREAKS): the v2
radiance formula is scale * (raw - offset) per PUG M1054E where the legacy
reader used scale * raw + offset, so compat.v1 raises NotImplementedError —
there is no v1 mapping (pinned by test_l1_v1_clean_break).
"""

import gc

import numpy as np
import pytest

from core.geo.naming import names

from . import generic
from .conftest import require_sample

from eoread.compat import v1
from eoread.reader import ProductError, UnsupportedProductType
from eoread.readers.modis import Level1B_MODIS


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles, so reader datasets are only freed
    # by the cyclic GC. Force a collection after every test to keep the
    # session memory bounded (see test_msi.py).
    yield
    gc.collect()


@pytest.fixture(scope='session')
def level1_modisa_myd():
    return require_sample('LEVEL1_MODISA_MYD')


@pytest.fixture(params=[200, (200, 300)])
def chunks(request):
    return request.param


@pytest.fixture
def l1(level1_modisa_myd, chunks):
    return Level1B_MODIS(level1_modisa_myd, chunks=chunks).ds


# ===========================================================================
# Level 1B (swath)
# ===========================================================================

def test_l1_instantiation(level1_modisa_myd, chunks):
    Level1B_MODIS(level1_modisa_myd, chunks=chunks)


def test_l1_main(l1):
    # Full-grid angles are not emitted for MODIS L1B (raw tie-point angles
    # only, on the y_tie/x_tie grid).
    generic.Test.main(l1, angle_data=False)


def test_l1_conformance(l1):
    generic.Test.v2_conformance(l1, primary='Ltoa')


def test_l1_latlon(l1):
    # The swath is a tilted parallelogram: no flat-projection invariance.
    generic.Test.latlon(l1, flat_proj=False)


def test_l1_lazy_load(l1):
    generic.Test.lazy_load(l1)


def test_l1_flagreader(l1):
    generic.Test.flagreader(l1)


def test_l1_subset(level1_modisa_myd, chunks):
    ds = Level1B_MODIS(level1_modisa_myd, chunks=chunks).ds
    generic.Test.subset(ds)


def test_l1_plot(request, level1_modisa_myd):
    ds = Level1B_MODIS(level1_modisa_myd).ds
    # Band labels are the file-native float values as strings (legacy parity).
    generic.plot(request, ds, '1.0', poi={'x': 1000, 'y': 1000})


def test_l1_time(chunks, level1_modisa_myd):
    params = {'path': level1_modisa_myd, 'chunks': chunks}
    generic.Test.execution_time(Level1B_MODIS, params)


def test_l1_bands(l1):
    # File block order: 250M [1, 2], 500M [3-7], 1KM_RefSB [8-19 with the
    # high-gain TDI pairs 13.5/14.5, plus 26], 1KM_Emissive [20-25, 27-36].
    expected = ([f'{b}.0' for b in (1, 2, 3, 4, 5, 6, 7)]
                + [f'{b}.0' for b in (8, 9, 10, 11, 12, 13)] + ['13.5']
                + ['14.0', '14.5'] + [f'{b}.0' for b in (15, 16, 17, 18, 19)]
                + ['26.0']
                + [f'{b}.0' for b in (20, 21, 22, 23, 24, 25)]
                + [f'{b}.0' for b in (27, 28, 29, 30, 31, 32, 33, 34, 35, 36)])
    assert list(l1.coords[str(names.bands)]) == expected
    # Band groups: the file-native block names, contiguous per block.
    bgroup = list(l1.coords[str(names.bgroup)].values)
    assert set(bgroup) == {'bands_250M', 'bands_500M',
                           'bands_1KM_RefSB', 'bands_1KM_Emissive'}
    assert bgroup.count('bands_250M') == 2
    assert bgroup.count('bands_500M') == 5
    assert bgroup.count('bands_1KM_RefSB') == 15
    assert bgroup.count('bands_1KM_Emissive') == 16


def test_l1_cwav(l1):
    cw = l1[str(names.cwav)]
    assert cw.attrs.get('unit') == 'nm'
    assert float(cw.sel({str(names.bands): '11.0'})) == pytest.approx(530.1)
    assert float(cw.sel({str(names.bands): '13.5'})) == pytest.approx(666.0)
    assert float(cw.sel({str(names.bands): '32.0'})) == pytest.approx(12042.3)


def test_l1_attrs(l1):
    assert l1.attrs[str(names.resolution)] == 1000
    assert l1.attrs[str(names.sensor)] == 'MODIS'
    assert l1.attrs[str(names.platform)] == 'Aqua'
    assert l1.attrs[str(names.shortname)] == 'MYD021KM'
    assert l1.attrs[str(names.product_name)] \
        == 'MYD021KM.A2018238.1740.006.2018239145651.hdf'
    assert l1.attrs[str(names.crs)] == 'EPSG:4326'
    # Acquisition start from the granule name (DOY 238 of 2018), not the
    # production datetime.
    assert l1.attrs[str(names.datetime)] == '2018-08-26T17:40:00+00:00'
    assert l1.attrs['night'] is False
    # SRF getter (REFACTORING.md decision log)
    assert l1.attrs['_srf_getter'] == 'eotools.srf.get_SRF_eumetsat'
    assert l1.attrs['_srf_getter_arg'] == 'eos_1_modis'


def test_l1_orientation(l1):
    """Canonical output is south-up, west-to-east (row 0 = southern edge)."""
    rows, cols = str(names.rows), str(names.columns)
    cx = l1.sizes[cols] // 2
    lat_first = float(l1[str(names.lat)].isel({rows: 0, cols: cx}).values)
    lat_last = float(l1[str(names.lat)].isel({rows: -1, cols: cx}).values)
    assert lat_first < lat_last
    cy = l1.sizes[rows] // 2
    lon_first = float(l1[str(names.lon)].isel({rows: cy, cols: 0}).values)
    lon_last = float(l1[str(names.lon)].isel({rows: cy, cols: -1}).values)
    assert lon_first < lon_last


def test_l1_rtoa_lazy(l1):
    """Rtoa computes lazily: reflective bands finite, emissive bands NaN."""
    rows, cols = str(names.rows), str(names.columns)
    win = {rows: slice(1000, 1020), cols: slice(900, 920)}
    # Band labels are strings; select each band by scalar label.
    r4 = l1[str(names.rtoa)].sel({str(names.bands): '4.0', **win}).values
    r32 = l1[str(names.rtoa)].sel({str(names.bands): '32.0', **win}).values
    assert np.any(np.isfinite(r4))     # band 4 (green) is reflective
    assert not np.any(np.isfinite(r32))  # band 32 is emissive
    ltoa = l1[str(names.ltoa)].sel({str(names.bands): '32.0', **win}).values
    # Physics check on the fixed formula scale * (raw - offset): band 32
    # brightness temperature is ~270 K over this scene, so Ltoa > 0.
    assert np.nanmedian(ltoa) > 0


def test_l1_tie_angles(l1):
    """Raw tie-point angles are kept on their own (y_tie, x_tie) grid."""
    ty, tx = str(names.rows) + '_tie', str(names.columns) + '_tie'
    for name in ('vza_tie', 'vaa_tie', 'sza_tie', 'saa_tie'):
        v = l1[name]
        assert v.dims == (ty, tx)
        assert v.dtype == np.float32
    # No full-grid angles are emitted.
    for name in ('vza', 'vaa', 'sza', 'saa'):
        assert str(getattr(names, name)) not in l1
    # Decimated range sanity check (never computes the full grid).
    sub = l1['sza_tie'].isel({ty: slice(None, None, 50),
                              tx: slice(None, None, 50)}).values
    assert np.nanmin(sub) >= 0 and np.nanmax(sub) <= 90


def test_l1_gflags(l1):
    """gflags is kept raw uint8 on the pixel grid, unmasked."""
    rows, cols = str(names.rows), str(names.columns)
    v = l1['gflags']
    assert v.dims == (rows, cols)
    assert v.dtype == np.uint8


def test_l1_v1_clean_break(level1_modisa_myd):
    ds = Level1B_MODIS(level1_modisa_myd).ds
    with pytest.raises(NotImplementedError):
        v1(ds)


def test_l1_pyhdf_fallback(level1_modisa_myd, chunks, monkeypatch):
    """The loud pyhdf fallback must produce a conforming dataset on real data."""
    import eoread.readers.modis as modis_module

    def fake_open(*args, **kwargs):
        raise OSError('[Errno -128] NetCDF: HDF4 is not supported')

    monkeypatch.setattr(modis_module.xr, 'open_dataset', fake_open)
    ds = Level1B_MODIS(level1_modisa_myd, chunks=chunks).ds
    generic.Test.main(ds, angle_data=False)


def test_l1_not_a_modis_product(tmp_path):
    with pytest.raises(ProductError):
        Level1B_MODIS(tmp_path)


def test_l1_rejects_other_shortname():
    header = {'INVENTORYMETADATA': {'COLLECTIONDESCRIPTIONCLASS':
              {'SHORTNAME': {'NUM_VAL': '1', 'VALUE': 'MCD43A4'}}}}
    # Pure detection logic: no file I/O, so skip __init__ entirely (the
    # method only reads the class-level VARIANTS/sensor/level attributes).
    reader = Level1B_MODIS.__new__(Level1B_MODIS)
    with pytest.raises(UnsupportedProductType):
        reader.detect_variant(header)
