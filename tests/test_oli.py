#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the Landsat OLI reader (Phase 5).

Samples (see tests/README.md):
- OLI_L1TP_L8: LC08 L1TP granule, EPSG:32610, 7671 x 7811 pixels at 30 m
  (pan B8 at 15341 x 15621 / 15 m).
- OLI_L1TP_L9: LC09 L1TP granule, EPSG:32621, 7831 x 7871 pixels at 30 m.

The legacy v1 reader is kept as the reference implementation
(eoread/legacy/oli.py); these tests pin the v2 contract instead: schema
conformance, canonical south-up orientation, lazy loading and flag decoding.
OLI has a documented clean break in eoread.compat (CLEAN_BREAKS): the v2 Rtoa
drops the legacy /cos(sza) double correction, so no v1 compatibility mapping
exists.
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.geo.naming import names

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.reader import ProductError, UnsupportedProductType
from eoread.readers.oli import Level1_OLI


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles, so reader datasets are only freed
    # by the cyclic GC. Force a collection after every test to keep the
    # session memory bounded (see test_msi.py).
    yield
    gc.collect()


@pytest.fixture(scope='session')
def level1_l8():
    return get_sample(Level1_OLI, 'OLI_L1TP_L8')


@pytest.fixture(scope='session')
def level1_l9():
    return get_sample(Level1_OLI, 'OLI_L1TP_L9')


@pytest.fixture(params=[500, (500, 700)])
def chunks(request):
    return request.param


@pytest.fixture(params=[15, 30])
def resolution(request):
    return request.param


@pytest.fixture
def l8(level1_l8, chunks, resolution):
    return Level1_OLI(level1_l8, chunks=chunks, resolution=resolution).ds


@pytest.fixture
def l9(level1_l9, chunks, resolution):
    return Level1_OLI(level1_l9, chunks=chunks, resolution=resolution).ds


################################################################################
# Level 1TP (Landsat-8)
################################################################################

def test_l8_instantiation(level1_l8, chunks, resolution):
    Level1_OLI(level1_l8, chunks=chunks, resolution=resolution)


def test_l8_main(l8):
    generic.Test.main(l8, angle_data=True)


def test_l8_conformance(l8):
    generic.Test.v2_conformance(l8, primary='Rtoa')


def test_l8_latlon(l8):
    generic.Test.latlon(l8, flat_proj=False)


def test_l8_angles(l8):
    generic.Test.angles(l8)


def test_l8_lazy_load(l8):
    generic.Test.lazy_load(l8)


def test_l8_flagreader(l8):
    generic.Test.flagreader(l8)


def test_l8_subset(level1_l8, chunks):
    ds = Level1_OLI(level1_l8, chunks=chunks, metadata_template=[]).ds
    generic.Test.subset(ds)


def test_l8_plot(request, level1_l8):
    # Full-grid compute; keep it on the 30 m grid.
    ds = Level1_OLI(level1_l8, resolution=30).ds
    generic.plot(request, ds, 'B4', poi={'x': 3000, 'y': 3000})


def test_l8_time(level1_l8, chunks):
    params = {'path': level1_l8, 'chunks': chunks}
    generic.Test.execution_time(Level1_OLI, params)


def test_l8_bands(l8):
    # All 11 bands on one grid; the pan band B8 is no longer split off.
    assert list(l8.coords['bands'].values) == \
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']
    groups = l8[str(names.bgroup)].values
    assert list(groups) == ['bands_30m'] * 7 + ['bands_15m'] + ['bands_30m'] * 3


def test_l8_cwav(l8):
    cw = l8[str(names.cwav)]
    assert cw.attrs.get('unit') == 'nm'
    assert cw.sel(bands='B1').values == pytest.approx(442.96)
    assert cw.sel(bands='B8').values == pytest.approx(590.0)
    assert cw.sel(bands='B9').values == pytest.approx(1373.43)
    assert cw.sel(bands='B11').values == pytest.approx(12050.0)


def test_l8_attrs(l8, resolution):
    assert l8.attrs[str(names.sensor)] == 'OLI'
    assert l8.attrs[str(names.platform)] == 'Landsat 8'
    assert l8.attrs[str(names.shortname)] == 'L1TP'
    assert l8.attrs[str(names.crs)] == 'EPSG:32610'
    assert l8.attrs[str(names.resolution)] == resolution
    assert l8.attrs['_srf_getter'] == 'eotools.srf.get_SRF_eumetsat'
    assert l8.attrs['_srf_getter_arg'] == 'landsat_8_oli'


def test_l8_orientation(l8):
    # South-up: latitude increases with row index.
    lat = l8[str(names.lat)].isel({str(names.columns): 0}).values
    assert lat[0] < lat[-1]


def test_l8_rtoa_bt(level1_l8):
    """Pin the USGS C2 calibration (no /cos(sza) division) on small windows."""
    ds = Level1_OLI(level1_l8, resolution=30).ds
    rows, cols = str(names.rows), str(names.columns)

    # A window that is entirely DN=0 in the source file (footprint edge, the
    # GDAL_NODATA value): it must be masked to NaN rather than calibrated to
    # the additive factor (-0.1), which would look like a plausible but false
    # negative reflectance.
    zero_win = {rows: slice(1000, 1020), cols: slice(900, 920)}
    r4z = ds[str(names.rtoa)].sel(bands='B4').isel(zero_win).values
    assert np.isnan(r4z).all()

    # A mid-scene window with real signal (DN ~ 10000 -> Rtoa ~ 0.1).
    win = {rows: slice(3990, 4010), cols: slice(3900, 3920)}
    r4 = ds[str(names.rtoa)].sel(bands='B4').isel(win).values
    assert np.isfinite(r4).all()
    assert r4.min() >= -0.1 - 1e-6
    assert 0.05 < r4.max() < 2.3

    r10 = ds[str(names.rtoa)].sel(bands='B10').isel(win).values
    assert np.isnan(r10).all()

    bt10 = ds[str(names.bt)].sel(bands='B10').isel(win).values
    assert np.isfinite(bt10).all()
    assert 140. < bt10.min() and bt10.max() < 350.

    bt4 = ds[str(names.bt)].sel(bands='B4').isel(win).values
    assert np.isnan(bt4).all()


def test_l8_qa_pixel(l8):
    qa = l8['QA_PIXEL']
    assert qa.dtype == np.uint16
    assert qa.dims == (str(names.rows), str(names.columns))


def test_l8_v1_clean_break(level1_l8):
    ds = Level1_OLI(level1_l8, resolution=30).ds
    with pytest.raises(NotImplementedError):
        v1(ds)


def test_l8_exit_clean_stderr(level1_l8, tmp_path):
    """A reader held until process exit must not emit teardown noise.

    Same guard as the MSI one (see test_msi.py): xarray's rasterio backend
    leaves one CachingFileManager open per file for lazy windowed reads; the
    atexit close installed by install_exit_cleanup (eoread.tools) must run
    before finalization or CPython >= 3.12 prints a nested 'Error in
    sys.excepthook' pair per open file (RC stays 0).
    """
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        "from eoread.readers.oli import Level1_OLI\n"
        f"reader = Level1_OLI({str(level1_l8)!r}, resolution=15)\n"
        "assert reader.ds is not None\n"
        "print('READER_BUILT')\n",
    )
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'READER_BUILT' in proc.stdout
    assert 'excepthook' not in proc.stderr, \
        'teardown noise at exit: file managers were not closed by atexit'


def test_l8_resolution_invalid(level1_l8):
    with pytest.raises(ValueError, match='unsupported resolution'):
        Level1_OLI(level1_l8, resolution=10)


def test_not_a_product(tmp_path):
    with pytest.raises(ProductError):
        Level1_OLI(tmp_path)


################################################################################
# Level 1TP (Landsat-9)
################################################################################
# The resampling/calibration code is mission-agnostic and covered above on
# L8; here we pin what actually differs per mission: variant detection, the
# platform attribute, the CRS and the OLI-2 wavelength table.

def test_l9_instantiation(level1_l9, chunks, resolution):
    Level1_OLI(level1_l9, chunks=chunks, resolution=resolution)


def test_l9_main(l9):
    generic.Test.main(l9, angle_data=True)


def test_l9_conformance(l9):
    generic.Test.v2_conformance(l9, primary='Rtoa')


def test_l9_attrs(l9):
    assert l9.attrs[str(names.platform)] == 'Landsat 9'
    assert l9.attrs[str(names.crs)] == 'EPSG:32621'
    assert l9.attrs['_srf_getter_arg'] == 'landsat_9_oli'


def test_l9_cwav(l9):
    cw = l9[str(names.cwav)]
    assert cw.sel(bands='B4').values == pytest.approx(655.0)
    assert cw.sel(bands='B9').values == pytest.approx(1375.0)


def test_rejects_unknown_spacecraft():
    reader = Level1_OLI.__new__(Level1_OLI)
    with pytest.raises(UnsupportedProductType):
        reader.detect_variant({'IMAGE_ATTRIBUTES': {'SPACECRAFT_ID': 'LANDSAT_7'}})
