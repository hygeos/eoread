#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance + legacy parity tests for the PlanetScope reader
(PSScene Basic Analytic 8b).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

Parity with the legacy reader (eoread.legacy.planetscope) is pinned at two
fixed windows of pixel coordinates (north-up rows/columns in the legacy
space). Documented deviations from legacy (see REFACTORING.md):

- south-up orientation: v2 row i carries legacy row ny - 1 - i; windows are
  therefore compared with the legacy rows flipped.
- nodata pixels (DN == 0) become NaN; the legacy rioxarray open kept them as
  zero data (masked=False default).
- `cwav` is float64 (legacy float32); the band labels are the GeoTIFF band
  descriptions, verbatim (as in legacy).
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rioxarray
from core.geo.naming import names
from core.table import read_xml

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.readers.planetscope import Level1_PLANETSCOPE
from eoread.legacy.planetscope import Level1_Planetscope as legacy_planetscope
from eoread.tools import xml_value

# Fixed comparison windows in north-up (file/legacy) rows/columns; both well
# inside the 5304 x 8880 scene.
WIN1 = (100, 200)
WIN2 = (2800, 2700)
WIN = 5

# float32 rounding tolerances (legacy mixed float64 intermediates).
PAYLOAD_RTOL = 1e-6
LATLON_ATOL = 1e-6
ANGLE_ATOL = 1e-7


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so transient legacy builds do not accumulate across the session.
    yield
    gc.collect()


@pytest.fixture(scope='session')
def product_dir():
    return get_sample(Level1_PLANETSCOPE)


@pytest.fixture(scope='session')
def ogc(product_dir):
    xml_paths = sorted(product_dir.glob('**/*.xml'))
    assert len(xml_paths) == 1, f'expected one XML in {product_dir}, got {xml_paths}'
    return read_xml(str(xml_paths[0]))


@pytest.fixture(scope='session')
def ds(product_dir):
    return Level1_PLANETSCOPE(product_dir, chunks=500)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _tif(product_dir):
    tifs = sorted(product_dir.glob('**/*8b.tif'))
    assert len(tifs) == 1, f'expected one *8b.tif in {product_dir}, got {tifs}'
    return tifs[0]


def _window_sels(ny, y0, x0):
    """(v2 south-up row slice, shared column slice) of a legacy window."""
    return (slice(ny - 1 - (y0 + WIN - 1), ny - y0), slice(x0, x0 + WIN))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_instantiation(product_dir):
    # default chunks
    Level1_PLANETSCOPE(product_dir)


def test_main(ds):
    generic.Test.main(ds, angle_data=True)


def test_conformance(ds):
    generic.Test.v2_conformance(ds, primary='Rtoa')


def test_latlon(ds):
    # Tilted scene: lat varies by more than 1e-7 deg
    generic.Test.latlon(ds, flat_proj=False)


def test_angles(ds):
    generic.Test.angles(ds)


def test_lazy_load(ds):
    generic.Test.lazy_load(ds)


def test_subset(product_dir):
    ds = Level1_PLANETSCOPE(product_dir, metadata_template=[])
    generic.Test.subset(ds)


def test_plot(request, ds):
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    # The 8b band labels are lowercase (GeoTIFF descriptions).
    generic.plot(request, ds, 'nir',
                 poi={str(names.rows): ny // 2, str(names.columns): nx // 2})


def test_bands_attrs(product_dir, ds, ogc):
    meta = ogc['metaDataProperty']['EarthObservationMetaData']
    platform = ogc['using']['EarthObservationEquipment']['platform']['Platform']

    # The band labels are the GeoTIFF descriptions verbatim (probe-verified).
    with rasterio.open(str(_tif(product_dir))) as src:
        descriptions = list(src.descriptions)
    pinned_bands = ['coastal_blue', 'blue', 'green_i', 'green',
                    'yellow', 'red', 'rededge', 'nir']
    assert descriptions == pinned_bands
    assert list(ds.coords[str(names.bands)].values) == pinned_bands
    assert list(ds.coords[str(names.bgroup)].values) == ['bands_vnir'] * 8

    assert ds.attrs[str(names.sensor)] == 'PLANETSCOPE'
    assert ds.attrs[str(names.platform)] == \
        str(xml_value(platform['shortName'])) == 'PlanetScope'
    assert ds.attrs[str(names.shortname)] == \
        str(xml_value(meta['productType'])) == 'L1B'
    assert ds.attrs[str(names.product_name)] == product_dir.name
    assert ds.attrs[str(names.datetime)] == str(xml_value(
        meta['downlinkedTo']['DownlinkInformation']['acquisitionDate']))
    # The reader stores the parent of the resolved product root.
    assert ds.attrs[str(names.input_directory)] == \
        str(product_dir.resolve().parent)

    resolution = ds.attrs[str(names.resolution)]
    assert resolution == 3 and isinstance(resolution, int)
    assert ds.attrs[str(names.crs)] == 'EPSG:4326'

    assert ds.attrs['_srf_getter'] == \
        'eoread.readers.planetscope.read_srf_superdove'
    assert ds.attrs['_srf_getter_arg'] is None

    # Central wavelengths are hardcoded in the reader (file band order).
    assert ds[str(names.cwav)].dtype == 'float64'
    np.testing.assert_array_equal(ds[str(names.cwav)].values,
                                  [443., 490., 531., 565.,
                                   610., 665., 705., 865.])


def test_v1_compat(ds):
    old = v1(ds)
    assert 'Rtoa' in old
    assert 'schema_version' not in old.attrs


def test_angles_ground_truth(ds, ogc):
    """The angles are constant over the scene: exact float32 values.

    Expected values mirror the reader's OGC lookups (scientific-notation
    strings unwrapped through xml_value).
    """
    acq = ogc['using']['EarthObservationEquipment'] \
        ['acquisitionParameters']['Acquisition']
    expected = {
        'sza': np.float32(90.0 - float(xml_value(acq['illuminationElevationAngle']))),
        'saa': np.float32(float(xml_value(acq['illuminationAzimuthAngle']))),
        'vza': np.float32(float(xml_value(acq['spaceCraftViewAngle']))),
        'vaa': np.float32(float(xml_value(acq['azimuthAngle']))),
    }
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for var, exp in expected.items():
        got = ds[var].isel(y=[0, ny // 2, ny - 1],
                           x=[0, nx // 2, nx - 1]).compute().values
        assert np.array_equal(got, np.full(got.shape, exp)), var


def test_latlon_rpc(ds, product_dir):
    """Independent RPC evaluation: own file parse + rasterio transformer.

    The reader evaluates the model at 0-based pixel indices (no +0.5), as in
    legacy; v2 row r carries file row ny - 1 - r.
    """
    from rasterio.rpc import RPC
    from rasterio.transform import RPCTransformer

    rpc_paths = sorted(product_dir.glob('**/*_RPC.TXT'))
    assert len(rpc_paths) == 1, f'expected one RPC file, got {rpc_paths}'

    # Independent KEY: value parse of the RPC text file.
    rpc_dict = {}
    for line in rpc_paths[0].read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key, value = line.split(':', 1)
            rpc_dict[key.strip()] = float(value.strip())

    rpc = RPC(
        line_off=rpc_dict['LINE_OFF'],
        line_scale=rpc_dict['LINE_SCALE'],
        line_num_coeff=[rpc_dict[f'LINE_NUM_COEFF_{i}'] for i in range(1, 21)],
        line_den_coeff=[rpc_dict[f'LINE_DEN_COEFF_{i}'] for i in range(1, 21)],
        samp_off=rpc_dict['SAMP_OFF'],
        samp_scale=rpc_dict['SAMP_SCALE'],
        samp_num_coeff=[rpc_dict[f'SAMP_NUM_COEFF_{i}'] for i in range(1, 21)],
        samp_den_coeff=[rpc_dict[f'SAMP_DEN_COEFF_{i}'] for i in range(1, 21)],
        lat_off=rpc_dict['LAT_OFF'],
        lat_scale=rpc_dict['LAT_SCALE'],
        long_off=rpc_dict['LONG_OFF'],
        long_scale=rpc_dict['LONG_SCALE'],
        height_off=rpc_dict['HEIGHT_OFF'],
        height_scale=rpc_dict['HEIGHT_SCALE'],
    )
    t = RPCTransformer(rpc)
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]

    # The WIN1 grid: transformer.xy returns (lon, lat), flattened.
    rows = np.arange(WIN1[0], WIN1[0] + WIN)
    cols = np.arange(WIN1[1], WIN1[1] + WIN)
    line_g, samp_g = np.meshgrid(rows, cols, indexing='ij')
    lon_e, lat_e = t.xy(line_g, samp_g, zs=0.0)
    lat_e = np.asarray(lat_e).reshape(line_g.shape)
    lon_e = np.asarray(lon_e).reshape(samp_g.shape)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            got_lat = ds['latitude'].isel(y=ny - 1 - r, x=c).values
            got_lon = ds['longitude'].isel(y=ny - 1 - r, x=c).values
            assert abs(got_lat - lat_e[i, j]) < 1e-5, (r, c)
            assert abs(got_lon - lon_e[i, j]) < 1e-5, (r, c)

    # A few extra points across the scene.
    for r, c in ((0, 0), (ny // 2, nx // 2), (ny - 1, nx - 1)):
        lon_e, lat_e = t.xy(np.array([r]), np.array([c]), zs=0.0)
        got_lat = ds['latitude'].isel(y=ny - 1 - r, x=c).values
        got_lon = ds['longitude'].isel(y=ny - 1 - r, x=c).values
        assert abs(got_lat - float(lat_e[0])) < 1e-5, (r, c)
        assert abs(got_lon - float(lon_e[0])) < 1e-5, (r, c)


def test_nodata_mask(product_dir, ds):
    """v2 NaN positions must equal the raw DN nodata flag on the edge lines.

    The file band order equals the v2 band order (identity map).
    """
    raw = rioxarray.open_rasterio(str(_tif(product_dir)), masked=False,
                                  chunks=500)
    raw = raw.isel(y=slice(None, None, -1)).rename(band='bands')  # north-up flip
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]

    lines = [
        {str(names.rows): 0},
        {str(names.rows): ny - 1},
        {str(names.columns): 0},
        {str(names.columns): nx - 1},
    ]
    for i, label in enumerate(ds.coords[str(names.bands)].values):
        for sel in lines:
            v2 = ds['Rtoa'].isel(bands=i, **sel).compute().values
            dn = raw.isel(bands=i, **sel).values
            assert np.array_equal(np.isnan(v2), (dn == 0.0)), (label, sel)


def _parity_window(ds, legacy, raw, ny, y0, x0):
    """Compare one 5x5 legacy window against the v2 south-up image of it.

    NaN positions are checked against the raw DN nodata flags (the legacy
    reader kept DN == 0 as zero data); values on the unmasked pixels only.
    The file band order equals the v2 band order (identity map).
    """
    jy, jx = _window_sels(ny, y0, x0)
    ly, lx = slice(y0, y0 + WIN), slice(x0, x0 + WIN)

    for i, label in enumerate(ds.coords[str(names.bands)].values):
        a = ds['Rtoa'].isel(bands=i).isel(y=jy, x=jx).compute().values
        b = legacy['Rtoa'].sel(bands=label).isel(y=ly, x=lx).compute().values[::-1]
        dn = raw.isel(bands=i, y=jy, x=jx).values
        m = ~np.isnan(a)
        assert np.array_equal(np.isnan(a), dn == 0.0), f'Rtoa {label} NaN mask'
        if m.any():
            assert np.allclose(a[m], b[m], rtol=PAYLOAD_RTOL, atol=0), \
                f'Rtoa {label} value mismatch'

    for var in ('latitude', 'longitude'):
        a = ds[var].isel(y=jy, x=jx).compute().values
        b = legacy[var].isel(y=ly, x=lx).compute().values[::-1]
        assert np.allclose(a, b, atol=LATLON_ATOL), var

    for var in ('sza', 'saa', 'vza', 'vaa'):
        a = ds[var].isel(y=jy, x=jx).compute().values
        b = legacy[var].isel(y=ly, x=lx).compute().values[::-1]
        assert np.allclose(a, b, atol=ANGLE_ATOL), var


def test_parity_legacy(ds, product_dir):
    """Value parity with the legacy reader at both fixed windows.

    The legacy reader evaluates the RPC model eagerly on the full grid
    (~2 GB transient); this is accepted once per session.
    """
    raw = rioxarray.open_rasterio(str(_tif(product_dir)), masked=False,
                                  chunks=500)
    raw = raw.isel(y=slice(None, None, -1)).rename(band='bands')  # north-up flip
    ny = ds.sizes[str(names.rows)]
    legacy = legacy_planetscope(product_dir, chunks=500)
    try:
        _parity_window(ds, legacy, raw, ny, *WIN1)
        _parity_window(ds, legacy, raw, ny, *WIN2)
    finally:
        del legacy
        gc.collect()


def test_exit_probe(product_dir, tmp_path):
    """A reader held until process exit must not emit teardown noise.

    rioxarray leaves one rasterio dataset open per lazily opened file; the
    atexit close (install_exit_cleanup) must release them before the
    interpreter finalizes, otherwise CPython >= 3.12 prints a nested
    'Error in sys.excepthook' pair for each of them.
    """
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        "from eoread.readers.planetscope import Level1_PLANETSCOPE\n"
        f"ds = Level1_PLANETSCOPE({str(product_dir)!r})\n"
        "assert ds is not None\n"
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


def test_missing_path():
    with pytest.raises(AssertionError):
        Level1_PLANETSCOPE(Path('/nonexistent/planetscope_product'))
