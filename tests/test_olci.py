#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the OLCI readers (Phase 3).

Samples (see tests/README.md):
- OLCI_L1_EFR: S3A OLCI L1 EFR granule, 4091 x 4865 pixels, 21 acquisition
  bands, low-latitude ocean scene.
- OLCI_L2_WFR: S3A OLCI L2 WFR water product (same frame grid, 16-band ocean
  subset).
- OLCI_L2_WRR: S3A OLCI L2 WRR water product (reduced resolution, 600 m).

The samples are local, so full-grid operations (plot, materialized checks)
are exercised. The v1 reader has been removed; these tests pin the v2
contract: schema conformance, canonical south-up orientation, lazy loading
and flag decoding.
"""

import gc
import os

import numpy as np
import pytest

from core.geo.naming import names

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.reader import ProductError, UnsupportedProductType
from eoread.readers.olci import Level1_OLCI, Level2_OLCI


def _check_angles(ds):
    """Sensor-native angle range check on a decimated sample (never computes
    the full grid).

    Deliberately not `generic.Test.angles`, which checks against the naming
    registry minv/maxv: OLCI stores SAA/VAA as signed degrees in [-180, 180]
    (Copernicus DSG), while the core registry declares [0, 180] for every
    azimuth — a registry inconsistency no ported sensor can satisfy (see
    REFACTORING.md).
    """
    ranges = {str(names.sza): (0., 90.), str(names.vza): (0., 90.),
              str(names.saa): (-180., 180.), str(names.vaa): (-180., 180.)}
    for name, (lo, hi) in ranges.items():
        assert name in ds, f'missing angle {name}'
        v = ds[name].isel({str(names.rows): slice(None, None, 50),
                           str(names.columns): slice(None, None, 50)}).values
        assert v.min() >= lo - 1e-3 and v.max() <= hi + 1e-3, \
            f'{name} out of range [{lo}, {hi}], got [{v.min()}, {v.max()}]'


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles, so reader datasets are only freed
    # by the cyclic GC. Force a collection after every test to keep the
    # session memory bounded (see test_msi.py).
    yield
    gc.collect()


def _readable(path):
    """Skip cleanly when a staged sample cannot be traversed/read.

    A .SEN3 granule is a directory tree: a missing execute bit on any level
    raises PermissionError at the first open instead of a useful test failure.
    """
    if not os.access(path, os.R_OK | os.X_OK):
        pytest.skip(f'sample not readable (check permissions): {path}')
    return path


@pytest.fixture(scope='session')
def level1_olci():
    return _readable(get_sample(Level1_OLCI))


@pytest.fixture(scope='session')
def level2_olci():
    return _readable(get_sample(Level2_OLCI, 'OLCI_L2_WFR'))


@pytest.fixture(scope='session')
def level2_olci_wrr():
    return _readable(get_sample(Level2_OLCI, 'OLCI_L2_WRR'))


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


@pytest.fixture
def l1(level1_olci, chunks):
    return Level1_OLCI(level1_olci, chunks=chunks)


@pytest.fixture
def l2(level2_olci, chunks):
    return Level2_OLCI(level2_olci, chunks=chunks)


@pytest.fixture
def l2_wrr(level2_olci_wrr, chunks):
    return Level2_OLCI(level2_olci_wrr, chunks=chunks)


# ===========================================================================
# Level 1 (EFR)
# ===========================================================================

def test_l1_instantiation(level1_olci, chunks):
    Level1_OLCI(level1_olci, chunks=chunks)


def test_l1_main(l1):
    generic.Test.main(l1, angle_data=True)


def test_l1_conformance(l1):
    generic.Test.v2_conformance(l1, primary='Ltoa')


def test_l1_latlon(l1):
    # The swath is a tilted parallelogram: no flat-projection invariance.
    generic.Test.latlon(l1, flat_proj=False)


def test_l1_angles(l1):
    _check_angles(l1)


def test_l1_lazy_load(l1):
    generic.Test.lazy_load(l1)


def test_l1_flagreader(l1):
    generic.Test.flagreader(l1)


def test_l1_subset(level1_olci, chunks):
    ds = Level1_OLCI(level1_olci, chunks=chunks)
    generic.Test.subset(ds)


def test_l1_plot(request, level1_olci):
    ds = Level1_OLCI(level1_olci)
    generic.plot(request, ds, 'Oa09', poi={'x': 1000, 'y': 3000})


def test_l1_time(chunks, level1_olci):
    params = {'path': level1_olci, 'chunks': chunks}
    generic.Test.execution_time(Level1_OLCI, params)


def test_l1_bands(l1):
    expected = [f'Oa{i:02d}' for i in range(1, 22)]
    assert list(l1.coords[str(names.bands)]) == expected


def test_l1_attrs(l1):
    assert l1.attrs[str(names.resolution)] == 300
    assert l1.attrs[str(names.sensor)] == 'OLCI'
    assert l1.attrs[str(names.platform)] == 'Sentinel-3A'
    assert l1.attrs[str(names.shortname)] == 'OL_1_EFR___'
    assert l1.attrs[str(names.crs)] == 'EPSG:4326'
    # SRF getter, loud on unknown platforms (REFACTORING.md decision log)
    assert l1.attrs['_srf_getter'] == 'eotools.srf.get_SRF_olci'
    assert l1.attrs['_srf_getter_arg'] == 'sentinel3_1_olci'


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
    """The declared extra Rtoa must compute lazily and stay consistent with Ltoa.

    Exercises the full per-pixel spectral chain on real data: tie-point SZA
    interpolation, the detector gather of the solar flux table and the lazy
    reflectance formula pi * Ltoa / (cos(sza) * F0).
    """
    rows, cols = str(names.rows), str(names.columns)
    win = {rows: slice(1500, 1520), cols: slice(2400, 2420)}
    ltoa = l1[str(names.ltoa)].isel({str(names.bands): [4], **win}).values
    rtoa = l1[str(names.rtoa)].isel({str(names.bands): [4], **win}).values
    assert np.all(np.isfinite(rtoa))
    mask = ltoa > 0
    assert mask.any()
    assert np.all(rtoa[mask] > 0)
    # Per-pixel wavelength of Oa05 (central wavelength ~510 nm): the detector
    # gather must land inside the band.
    wav = float(l1[str(names.wav)].isel({str(names.bands): [4], **win}).mean())
    assert 490 < wav < 530


def test_l1_v1_compat(l1):
    old = v1(l1)
    # OLCI has no documented clean break: the mapping is an identity apart
    # from the schema version, which v1 datasets did not carry.
    assert 'schema_version' not in old.attrs
    assert str(names.ltoa) in old
    assert str(names.rtoa) in old
    assert 'schema_version' in l1.attrs


def test_l1_rejects_level2(level2_olci):
    with pytest.raises(UnsupportedProductType):
        Level1_OLCI(level2_olci)


# ===========================================================================
# Level 2 (WFR / WRR water products)
# ===========================================================================

#: Ocean colour band subset of the OLCI L2 products (manifest order).
_L2_BANDS = [f'Oa{i:02d}' for i in list(range(1, 13)) + [16, 17, 18, 21]]


def test_l2_instantiation(level2_olci, chunks):
    Level2_OLCI(level2_olci, chunks=chunks)


def test_l2_main(l2):
    # rho_w is the spectral payload here (L2 water products carry no
    # ltoa/rtoa/bt), so it takes the primary role in check_vars.
    generic.Test.main(l2, angle_data=True, primary=str(names.rho_w))


def test_l2_conformance(l2):
    generic.Test.v2_conformance(l2, primary='rho_w')


def test_l2_latlon(l2):
    generic.Test.latlon(l2, flat_proj=False)


def test_l2_angles(l2):
    _check_angles(l2)


def test_l2_lazy_load(l2):
    generic.Test.lazy_load(l2)


def test_l2_subset(level2_olci, chunks):
    ds = Level2_OLCI(level2_olci, chunks=chunks)
    generic.Test.subset(ds)


def test_l2_plot(request, level2_olci):
    ds = Level2_OLCI(level2_olci)
    generic.plot(request, ds, 'Oa11', poi={'x': 1000, 'y': 3000})


def test_l2_time(chunks, level2_olci):
    params = {'path': level2_olci, 'chunks': chunks}
    generic.Test.execution_time(Level2_OLCI, params)


def test_l2_bands(l2):
    assert list(l2.coords[str(names.bands)]) == _L2_BANDS


def test_l2_attrs(l2):
    assert l2.attrs[str(names.resolution)] == 300
    assert l2.attrs[str(names.shortname)] == 'OL_2_WFR___'
    # The WQSF bit definitions are not in the core registry: no flag reader.
    assert '_flag_reader' not in l2.attrs


def test_l2_orientation(l2):
    rows, cols = str(names.rows), str(names.columns)
    cx = l2.sizes[cols] // 2
    lat_first = float(l2[str(names.lat)].isel({rows: 0, cols: cx}).values)
    lat_last = float(l2[str(names.lat)].isel({rows: -1, cols: cx}).values)
    assert lat_first < lat_last


def test_l2_v1_compat(l2):
    old = v1(l2)
    assert 'schema_version' not in old.attrs
    assert str(names.rho_w) in old


# ===========================================================================
# Level 2 WRR variant (reduced resolution)
# ===========================================================================

def test_l2wrr_main(l2_wrr):
    generic.Test.main(l2_wrr, angle_data=True, primary=str(names.rho_w))


def test_l2wrr_conformance(l2_wrr):
    generic.Test.v2_conformance(l2_wrr, primary='rho_w')


def test_l2wrr_resolution(l2_wrr):
    assert l2_wrr.attrs[str(names.resolution)] == 600
    assert l2_wrr.attrs[str(names.shortname)] == 'OL_2_WRR___'


def test_l2wrr_bands(l2_wrr):
    assert list(l2_wrr.coords[str(names.bands)]) == _L2_BANDS


# ===========================================================================
# Rejection of foreign inputs
# ===========================================================================

def test_l2_rejects_level1(level1_olci):
    with pytest.raises(UnsupportedProductType):
        Level2_OLCI(level1_olci)


def test_not_an_olci_product(tmp_path):
    with pytest.raises(ProductError):
        Level1_OLCI(tmp_path)
