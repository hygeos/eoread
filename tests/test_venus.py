#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance + legacy parity tests for the VENUS readers (L1C and L2A).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

Parity with the legacy readers (eoread.legacy.venus) is pinned at a fixed
window of pixel coordinates, plus full-array compares for the small tie-point
rasters. Documented deviations from legacy (see REFACTORING.md):

- south-up orientation: v2 row i carries legacy row ny - 1 - i; lat/lon are
  therefore compared at the *same* index (legacy stored them flipped relative
  to its own rasters).
- canonical tie-point angles (sza/saa/vza/vaa) are new; legacy computed none.
  The raw 2x2 XML grids are kept as ``*_tie`` variables.
- SOL_ALL / VIE_ALL are kept raw on their native tie grids. Legacy's assignment
  reindexed VIE onto SOL's smaller indexes: every tie point outside that grid
  (row 23, col 26) came back all-NaN, and the file's extra detector bands
  (bands 5-8 of 8) were dropped.
- L2A product nodata (-10000) becomes NaN; legacy kept -10.0 after calibration.
- L2A ATB bands are calibrated with the MTD quantification factors; legacy
  kept raw counts.
- L2A PIX_XS is assigned positionally; the legacy int band index vs string
  coordinate alignment silently produced an all-NaN variable.
- lat/lon come from a proper per-pixel UTM->WGS84 transform; legacy used a
  flipped linear interpolation between the granule corners.
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pyproj import Transformer
from core.geo.naming import names
from core.table import read_xml

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.flags import FlagsInit, GenericFlags
from eoread.reader import UnsupportedProductType
from eoread.readers.venus import (Level1C_VENUS, Level2A_VENUS, _find_file,
                                  _grid64, _grid_node, _val)
from eoread.legacy.venus import Level1_VENUS, Level2_VENUS

# Fixed comparison window: north-up raster rows x columns, well inside the
# 11405 x 12787 granule and away from the tie-raster NaN corners.
WIN_ROWS = np.array([4987, 5557, 6127])
WIN_COLS = np.array([5600, 6560, 7520])


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so transient legacy builds do not accumulate across the session.
    yield
    gc.collect()


@pytest.fixture(scope='session')
def l1_dir():
    return get_sample(Level1C_VENUS)


@pytest.fixture(scope='session')
def l2_dir():
    return get_sample(Level2A_VENUS)


@pytest.fixture(scope='session')
def l1(l1_dir):
    return Level1C_VENUS(l1_dir, chunks=500).ds


@pytest.fixture(scope='session')
def l2(l2_dir):
    return Level2A_VENUS(l2_dir, chunks=500).ds


@pytest.fixture(scope='session')
def l1_mtd(l1_dir):
    return read_xml(_find_file(l1_dir, '*MTD_ALL.xml'))


@pytest.fixture(scope='session')
def l2_mtd(l2_dir):
    return read_xml(_find_file(l2_dir, '*MTD_ALL.xml'))


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _grid(mtd):
    """The granule grid node of the MTD geopositioning."""
    return _grid_node(mtd['Geoposition_Informations']['Geopositioning']
                      ['Group_Geopositioning_List']['Group_Geopositioning'])


def _bilinear(grid, rows, cols, ny, nx):
    """Bilinear interpolation of a 2x2 tie grid at pixel corners.

    `grid` row 0 / col 0 is the north / west corner; `rows` and `cols` are
    north-up pixel indices of a ny x nx granule. float64 throughout.
    """
    v = rows.astype('float64')[:, None] / (ny - 1)
    u = cols.astype('float64')[None, :] / (nx - 1)
    north = (1 - u) * grid[0, 0] + u * grid[0, 1]
    south = (1 - u) * grid[1, 0] + u * grid[1, 1]
    return (1 - v) * north + v * south


def _angles_ground_truth_check(ds, mtd):
    """Check the canonical angles against an in-test bilinear of the raw XML grids."""
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    geo = mtd['Geometric_Informations']['Angles_Grids_List']
    sun = geo['Sun_Angles_Grids']
    view = (geo['Viewing_Incidence_Angles_Grids_List']
            ['Band_Viewing_Incidence_Angles_Grids_List']
            ['Viewing_Incidence_Angles_Grids'])
    grids = {
        'sza': _grid64(sun['Zenith']),
        'saa': _grid64(sun['Azimuth']),
        # the reader averages the four detector grids (float64) before storing
        'vza': np.mean([_grid64(g['Zenith']) for g in view], axis=0),
        'vaa': np.mean([_grid64(g['Azimuth']) for g in view], axis=0),
    }
    rows = np.array([0, ny // 4, ny - 1])
    cols = np.array([0, nx // 4, nx - 1])
    for name, grid in grids.items():
        want = _bilinear(grid, rows, cols, ny, nx)
        got = ds[name].isel(y=(ny - 1 - rows).tolist(), x=cols.tolist()).compute().values
        assert np.allclose(got, want, atol=1e-4), (name, got.ravel(), want.ravel())


def _latlon_pyproj_check(ds, mtd):
    """Independent UTM->WGS84 check of the lat/lon rasters with pyproj.

    Pixel centers are transformed directly from the MTD grid; this does not
    reuse any eoread geometry code path.
    """
    ny = ds.sizes[str(names.rows)]
    g = _grid(mtd)
    ulx, uly = float(g['ULX']), float(g['ULY'])
    xdim, ydim = float(g['XDIM']), float(g['YDIM'])
    X = ulx + xdim / 2 + xdim * WIN_COLS
    Y = uly + ydim / 2 + ydim * WIN_ROWS
    xx, yy = np.meshgrid(X, Y)
    lon, lat = Transformer.from_crs('EPSG:32630', 'EPSG:4326', always_xy=True).transform(xx, yy)
    j = (ny - 1 - WIN_ROWS).tolist()
    c = WIN_COLS.tolist()
    got_lat = ds['latitude'].isel(y=j, x=c).compute().values
    got_lon = ds['longitude'].isel(y=j, x=c).compute().values
    assert np.allclose(got_lat, lat, atol=1e-5), (got_lat.ravel(), lat.ravel())
    assert np.allclose(got_lon, lon, atol=1e-5), (got_lon.ravel(), lon.ravel())


def _latlon_corners_check(ds, mtd):
    """Check the v2 corner pixels against the MTD Global_Geopositioning corners."""
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    corners = (mtd['Geoposition_Informations']['Geopositioning']
               ['Global_Geopositioning'])
    # v2 is south-up: row 0 is the southern edge of the granule
    for (r, c), key in (((0, 0), 'lowerLeft'), ((0, nx - 1), 'lowerRight'),
                        ((ny - 1, 0), 'upperLeft'), ((ny - 1, nx - 1), 'upperRight')):
        lat = ds['latitude'].isel(y=r, x=c).compute().values
        lon = ds['longitude'].isel(y=r, x=c).compute().values
        assert abs(lat - corners[key]['LAT']) < 5e-4, (key, 'lat', lat)
        assert abs(lon - corners[key]['LON']) < 5e-4, (key, 'lon', lon)


def _parity_angle_rasters(ds, legacy):
    """Compare the SOL_ALL / VIE_ALL tie rasters against the legacy reader."""
    # SOL: both readers keep the raw 4x24x27 tie raster -> bit equality.
    a, b = ds['SOL_ALL'].compute(), legacy['SOL_ALL'].compute()
    assert a.sizes == b.sizes
    assert np.array_equal(a.values, b.values, equal_nan=True)

    # VIE: v2 keeps the raw 8x116x129 raster on its native tie grid; the legacy
    # reader's assignment aligned it onto SOL's indexes and lost both every tie
    # point outside SOL's 24x27 grid (row 23 / col 26 came back all-NaN) and the
    # file's extra detector bands (bands 5-8 dropped). Compare where the legacy
    # grid coincides with v2's native points, on the shared band subset.
    a, b = ds['VIE_ALL'].compute(), legacy['VIE_ALL'].compute()
    nb = b.sizes[b.dims[0]]
    assert a.sizes[a.dims[0]] >= nb
    a = a.isel({a.dims[0]: slice(None, nb)})
    # Check on the legacy (reindexed) grid: each of its 24x27 points either
    # coincides exactly with a v2 native VIE point (value must match) or has no
    # tie point at all and must be NaN there.
    dr_b = np.abs(b['y_tie'].values[:, None] - a['y_view'].values[None, :])
    dc_b = np.abs(b['x_tie'].values[:, None] - a['x_view'].values[None, :])
    jr_b = dr_b.argmin(axis=1)
    ic_b = dc_b.argmin(axis=1)
    ok_b = (dr_b.min(axis=1) == 0.0)[:, None] & (dc_b.min(axis=1) == 0.0)[None, :]
    want = a.values[np.ix_(np.arange(nb), jr_b, ic_b)]
    assert ok_b.any(), 'no legacy VIE tie point coincides with the v2 native grid'
    assert np.array_equal(b.values[:, ok_b], want[:, ok_b], equal_nan=True)
    assert np.isnan(b.values[:, ~ok_b]).all(), \
        'legacy VIE should be NaN where it has no tie point (SOL row 23 / col 26)'


def _window(ds):
    """(v2 y indices, legacy/north-up rows, shared columns) for the window."""
    ny = ds.sizes[str(names.rows)]
    return (ny - 1 - WIN_ROWS).tolist(), WIN_ROWS.tolist(), WIN_COLS.tolist()


# --------------------------------------------------------------------------
# Level-1C
# --------------------------------------------------------------------------

def test_l1_instantiation(l1_dir):
    # default chunks
    Level1C_VENUS(l1_dir)


def test_l1_main(l1):
    generic.Test.main(l1, angle_data=True)


def test_l1_conformance(l1):
    generic.Test.v2_conformance(l1, primary='Rtoa')


def test_l1_latlon(l1):
    # UTM tile: lat varies along a row by ~2e-6 deg, so not a flat projection
    generic.Test.latlon(l1, flat_proj=False)


def test_l1_lazy_load(l1):
    generic.Test.lazy_load(l1)


def test_l1_masks_lazy(l1):
    """The zipped masks must be windowed reads, not eager full-array loads.

    `generic.Test.lazy_load` only checks that a dask array is present; a
    `.chunk()` call on an already-materialized numpy array also passes that
    check.  This inspects the graph itself: no full-size numpy array may be
    embedded in it, and the source must stay a lazy raster reader.
    """
    ny = l1.sizes[str(names.rows)]
    nx = l1.sizes[str(names.columns)]
    for var in ('CLD_XS', 'USI_XS', 'PIX_XS', 'SAT_XS'):
        arr = l1[var]
        assert arr.chunks is not None, f'{var} is not lazy'
        for value in arr.data.__dask_graph__().values():
            if isinstance(value, np.ndarray):
                assert value.size < ny * nx, \
                    f'{var} embeds a full-grid numpy array in its graph'


def test_l1_flagreader(l1):
    generic.Test.flagreader(l1)


def test_l1_flag_values(l1):
    """Flag bits must reproduce the reader conditions on a 100x100 slice."""
    sub = l1.isel(y=slice(500, 600), x=slice(500, 600))
    flags = FlagsInit({GenericFlags.CLOUD: 1, GenericFlags.L1_INVALID: 4}, 'uint8',
                      l1.attrs['_flag_reader']).map_blocks(sub)
    f = flags['flags'].compute()
    assert np.array_equal((f & 1).astype(bool), (sub['CLD_XS'] != 0).compute().values)
    assert np.array_equal((f & 4).astype(bool), sub['CLA_ALL'].isnull().compute().values)


def test_l1_subset(l1_dir):
    ds = Level1C_VENUS(l1_dir, metadata_template=[]).ds
    generic.Test.subset(ds)


def test_l1_plot(request, l1):
    ny, nx = l1.sizes[str(names.rows)], l1.sizes[str(names.columns)]
    generic.plot(request, l1, 'B8', poi={str(names.rows): ny // 2, str(names.columns): nx // 2})


def test_l1_bands_attrs(l1, l1_mtd):
    assert list(l1.coords[str(names.bands)].values) == [f'B{i}' for i in range(1, 13)]
    assert list(l1.coords[str(names.bgroup)].values) == ['bands_vnir'] * 12
    assert l1.attrs[str(names.crs)] == 'EPSG:32630'
    pid = str(_val(l1_mtd['Product_Characteristics']['PRODUCT_ID']))
    assert l1.attrs[str(names.shortname)] == pid
    assert l1.attrs[str(names.product_name)] == pid
    assert l1.attrs[str(names.resolution)] == 5
    assert l1.attrs['_srf_getter'] == 'eoread.readers.venus.get_SRF'


def test_l1_v1_compat(l1):
    old = v1(l1)
    assert 'Rtoa' in old and 'SOL_ALL' in old
    assert 'schema_version' not in old.attrs


def test_l1_angles_ground_truth(l1, l1_mtd):
    _angles_ground_truth_check(l1, l1_mtd)


def test_l1_latlon_pyproj(l1, l1_mtd):
    _latlon_pyproj_check(l1, l1_mtd)


def test_l1_latlon_corners(l1, l1_mtd):
    _latlon_corners_check(l1, l1_mtd)


def test_l1_parity_legacy(l1, l1_dir):
    """Payload bit-parity with the legacy reader at the fixed window."""
    legacy = Level1_VENUS(l1_dir, chunks=500, verbose=False)
    try:
        j, r, c = _window(l1)

        # Rtoa: all 12 bands, bit-equal (same lazy open, same float32 division)
        a = l1['Rtoa'].isel(y=j, x=c).compute().values
        b = legacy['Rtoa'].isel(y=r, x=c).compute().values
        assert np.array_equal(a, b, equal_nan=True)

        for var in ('CLA_ALL', 'CLD_XS', 'USI_XS'):
            a = l1[var].isel(y=j, x=c).compute().values
            b = legacy[var].isel(y=r, x=c).compute().values
            assert np.array_equal(a, b, equal_nan=True), var

        # the legacy reader merges the per-band zips onto bare 'PIX' / 'SAT'
        for label in ('B1', 'B6', 'B12'):
            for var, leg_name in (('PIX_XS', 'PIX'), ('SAT_XS', 'SAT')):
                a = l1[var].sel(bands=label).isel(y=j, x=c).compute().values
                b = legacy[leg_name].sel(bands=label).isel(y=r, x=c).compute().values
                assert np.array_equal(a, b, equal_nan=True), (var, label)

        _parity_angle_rasters(l1, legacy)

        # lat/lon: legacy stored them flipped relative to its own rasters, so
        # v2[i] and legacy[i] carry the same geographic point at index i. The
        # difference is UTM transform vs corner linear interpolation (~1e-3 deg
        # of east-west swing across the tile).
        for var in ('latitude', 'longitude'):
            a = l1[var].isel(y=j, x=c).compute().values
            b = legacy[var].isel(y=j, x=c).compute().values
            assert np.allclose(a, b, atol=5e-3), var
    finally:
        del legacy
        gc.collect()


def test_l1_exit_probe(l1_dir, tmp_path):
    """A reader held until process exit must not emit teardown noise.

    rioxarray leaves one rasterio dataset open per lazily opened file; the
    atexit close (install_exit_cleanup) must release them before the
    interpreter finalizes, otherwise CPython >= 3.12 prints a nested
    'Error in sys.excepthook' pair for each of them.
    """
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        "from eoread.readers.venus import Level1C_VENUS\n"
        f"reader = Level1C_VENUS({str(l1_dir)!r})\n"
        "assert reader.ds is not None\n"
        "print('READER_BUILT')\n",
    )
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'READER_BUILT' in proc.stdout
    assert 'excepthook' not in proc.stderr, \
        'teardown noise at exit: lazy raster datasets were not closed by atexit'


def test_foreign_product_guard(l1_dir, l2_dir):
    with pytest.raises(UnsupportedProductType):
        Level1C_VENUS(l2_dir)
    with pytest.raises(UnsupportedProductType):
        Level2A_VENUS(l1_dir)


def test_missing_path():
    with pytest.raises(AssertionError):
        Level1C_VENUS(Path('/nonexistent/venus_product'))


# --------------------------------------------------------------------------
# Level-2A
# --------------------------------------------------------------------------

def test_l2_instantiation(l2_dir, chunks):
    Level2A_VENUS(l2_dir, chunks=chunks)


def test_l2_main(l2):
    generic.Test.main(l2, angle_data=True, primary='rho_surface')


def test_l2_conformance(l2):
    generic.Test.v2_conformance(l2, primary='rho_surface')


def test_l2_latlon(l2):
    generic.Test.latlon(l2, flat_proj=False)


def test_l2_lazy_load(l2):
    generic.Test.lazy_load(l2)


def test_l2_flagreader(l2):
    generic.Test.flagreader(l2)


def test_l2_flag_values(l2):
    """Flag bits must reproduce the reader conditions on a 100x100 slice."""
    sub = l2.isel(y=slice(500, 600), x=slice(500, 600))
    flags = FlagsInit({GenericFlags.CLOUD: 1, GenericFlags.L1_INVALID: 4}, 'uint8',
                      l2.attrs['_flag_reader']).map_blocks(sub)
    f = flags['flags'].compute()
    assert np.array_equal((f & 1).astype(bool), (sub['CLM_XS'] > 0).compute().values)
    assert np.array_equal(
        (f & 4).astype(bool),
        sub['rho_surface'].isnull().any(dim=str(names.bands)).compute().values)


def test_l2_subset(l2_dir, chunks):
    ds = Level2A_VENUS(l2_dir, chunks=chunks, metadata_template=[]).ds
    generic.Test.subset(ds)


def test_l2_bands_attrs(l2, l2_mtd):
    assert list(l2.coords[str(names.bands)].values) == [f'B{i}' for i in range(1, 13)]
    assert list(l2.coords[str(names.bgroup)].values) == ['bands_vnir'] * 12
    assert l2.attrs[str(names.crs)] == 'EPSG:32630'
    pid = str(_val(l2_mtd['Product_Characteristics']['PRODUCT_ID']))
    assert l2.attrs[str(names.shortname)] == pid
    assert l2.attrs[str(names.product_name)] == pid
    assert l2.attrs[str(names.resolution)] == 4


def test_l2_v1_compat(l2):
    old = v1(l2)
    assert 'rho_surface' in old and 'rho_flat' in old
    assert 'schema_version' not in old.attrs


def test_l2_angles_ground_truth(l2, l2_mtd):
    _angles_ground_truth_check(l2, l2_mtd)


def test_l2_latlon_pyproj(l2, l2_mtd):
    _latlon_pyproj_check(l2, l2_mtd)


def test_l2_latlon_corners(l2, l2_mtd):
    _latlon_corners_check(l2, l2_mtd)


def test_l2_parity_legacy(l2, l2_dir, l2_mtd):
    """Payload parity with the legacy reader at the fixed window."""
    # the legacy L2A entry point takes no `verbose` argument (L1C does)
    legacy = Level2_VENUS(l2_dir, chunks=500)
    try:
        j, r, c = _window(l2)
        radio = l2_mtd['Radiometric_Informations']
        q_wv = float(_val(radio['WATER_VAPOR_CONTENT_QUANTIFICATION_VALUE']))
        q_aot = float(_val(radio['AEROSOL_OPTICAL_THICKNESS_QUANTIFICATION_VALUE']))

        # Surface reflectances: bit-equal after mapping the legacy nodata
        # sentinel (-10000 -> -10.0 post-calibration) to NaN.
        for var in ('rho_surface', 'rho_flat'):
            a = l2[var].isel(y=j, x=c).compute().values
            b = legacy[var].isel(y=r, x=c).compute().values
            b[b == -10.0] = np.nan
            assert np.array_equal(a, b, equal_nan=True), var

        # water_vapor / aod: v2 calibrates with the MTD quantification factors,
        # legacy kept raw counts; bit-equal after the same float32 division.
        for var, q in (('water_vapor', q_wv), ('aod', q_aot)):
            a = l2[var].isel(y=j, x=c).compute().values
            b = legacy[var].isel(y=r, x=c).compute().values
            assert np.array_equal(a, (b / q).astype('float32')), var

        for var in ('CLM_XS', 'USI_XS', 'SAT_XS', 'IAB_XS', 'EDG_XS'):
            a = l2[var].isel(y=j, x=c).compute().values
            b = legacy[var].isel(y=r, x=c).compute().values
            assert np.array_equal(a, b, equal_nan=True), var

        # PIX_XS: pin against the raw file (positionally onto the MTD band
        # order); the legacy reader comes back all-NaN here (int vs string band
        # alignment bug) — asserted explicitly to document the deviation.
        pix = xr.open_dataarray(
            str(_find_file(l2_dir / 'MASKS', '*PIX_XS.tif')), engine='rasterio',
            chunks={'y': 500, 'x': 500})
        b = pix.isel(band=np.arange(12), y=r, x=c).compute().values
        a = l2['PIX_XS'].isel(y=j, x=c).compute().values
        assert np.array_equal(a, b, equal_nan=True)
        leg_pix = legacy['PIX_XS'].isel(y=r, x=c).compute().values
        assert np.isnan(leg_pix).all(), \
            'legacy PIX_XS is expected to be all-NaN (documented bug)'

        _parity_angle_rasters(l2, legacy)

        for var in ('latitude', 'longitude'):
            a = l2[var].isel(y=j, x=c).compute().values
            b = legacy[var].isel(y=j, x=c).compute().values
            assert np.allclose(a, b, atol=5e-3), var
    finally:
        del legacy
        gc.collect()


def test_l2_exit_probe(l2_dir, tmp_path):
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        "from eoread.readers.venus import Level2A_VENUS\n"
        f"reader = Level2A_VENUS({str(l2_dir)!r})\n"
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
        'teardown noise at exit: lazy raster datasets were not closed by atexit'
