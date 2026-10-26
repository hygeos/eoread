#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the GEOSAT readers (DE-1 and DE-2).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

One reader class serves both satellites; every test is parametrized over the
two samples. Documented deviations from the v1 implementation (see
REFACTORING.md):

- south-up orientation: v2 row i carries file row ny - 1 - i.
- GEOSAT-2 nodata pixels (DN == 0) become NaN; a plain rioxarray open keeps
  them as zero data (masked=False default), which would survive calibration.
- DE-2 `datetime` is the full START_TIME (v1 produced a date-only value);
  `cwav` is float64 (v1 int32).
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
from pyproj import Proj

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.readers.geosat import Level1_GEOSAT
from eoread.tools import _num, xml_value

# Fixed check window in north-up (file) rows/columns; well inside the
# 16200 x 7200 (DE-1) and 5676 x 5400 (DE-2) rasters.
WIN1 = (100, 200)


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so reader datasets are released promptly.
    yield
    gc.collect()


@pytest.fixture(scope='session', params=['GEOSAT_DE1', 'GEOSAT_DE2'],
                ids=['DE-1', 'DE-2'])
def product_dir(request):
    return get_sample(Level1_GEOSAT, request.param)


@pytest.fixture(scope='session')
def dim(product_dir):
    dims = sorted(product_dir.glob('*.dim'))
    assert len(dims) == 1, f'expected one .dim in {product_dir}, got {dims}'
    return read_xml(str(dims[0]))


@pytest.fixture(scope='session')
def ds(product_dir):
    return Level1_GEOSAT(product_dir, chunks=500)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _variant(ds):
    return ds.attrs[str(names.platform)]


def _tif(product_dir):
    tifs = sorted(product_dir.glob('*.tif'))
    assert len(tifs) == 1, f'expected one .tif in {product_dir}, got {tifs}'
    return tifs[0]


def _file_bands(dim):
    info = dim['Image_Interpretation']['Spectral_Band_Info']
    if isinstance(info, dict):
        info = [info]
    return [str(s['BAND_DESCRIPTION']) for s in info]


def _raw_band_map(product_dir, dim):
    """{band label: (south-up raw DN DataArray, in-file band index)}."""
    file_bands = _file_bands(dim)
    raw = rioxarray.open_rasterio(str(_tif(product_dir)), masked=False,
                                  chunks=500)
    raw = raw.isel(y=slice(None, None, -1)).rename(band='bands')  # north-up flip
    return {label: (raw, j) for j, label in enumerate(file_bands)}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_instantiation(product_dir):
    # default chunks
    Level1_GEOSAT(product_dir)


def test_main(ds):
    generic.Test.main(ds, angle_data=True)


def test_conformance(ds):
    generic.Test.v2_conformance(ds, primary='Ltoa')


def test_latlon(ds):
    # Tilted / gridded scenes: lat varies by more than 1e-7 deg
    generic.Test.latlon(ds, flat_proj=False)


def test_angles(ds):
    generic.Test.angles(ds)


def test_lazy_load(ds):
    generic.Test.lazy_load(ds)


def test_subset(product_dir):
    ds = Level1_GEOSAT(product_dir, metadata_template=[])
    generic.Test.subset(ds)


def test_plot(request, ds):
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    generic.plot(request, ds, 'NIR',
                 poi={str(names.rows): ny // 2, str(names.columns): nx // 2})


def test_bands_attrs(product_dir, ds, dim):
    variant = _variant(ds)
    scene = dim['Dataset_Sources']['Source_Information']['Scene_Source']

    # Band order recomputed independently: stable ascending sort of the DIMAP
    # file-order band descriptions by central wavelength.
    wave = {'NIR': 830, 'Red': 660, 'RED': 660, 'Green': 560, 'GREEN': 560,
            'Blue': 490, 'BLUE': 490}
    file_bands = _file_bands(dim)
    cwav_file = np.array([wave[b] for b in file_bands], dtype='float64')
    order = np.argsort(cwav_file, kind='stable')
    expected_bands = [file_bands[i] for i in order]
    expected_cwav = cwav_file[order]
    assert file_bands != expected_bands, 'expected the file order to be shuffled'

    # Pinned sample values (probe-verified).
    pinned = {
        'DE-1': (['Green', 'Red', 'NIR'], [560., 660., 830.]),
        'DE-2': (['BLUE', 'GREEN', 'RED', 'NIR'], [490., 560., 660., 830.]),
    }[variant]
    assert list(ds.coords[str(names.bands)].values) == expected_bands == pinned[0]
    assert list(ds.coords[str(names.bgroup)].values) == \
        ['bands_vnir'] * len(expected_bands)

    assert ds.attrs[str(names.sensor)] == 'GEOSAT'
    assert ds.attrs[str(names.platform)] == variant
    assert ds.attrs[str(names.shortname)] == str(dim['Production']['PRODUCT_TYPE'])
    assert {'DE-1': 'L1R', 'DE-2': 'L1C'}[variant] == \
        ds.attrs[str(names.shortname)]
    assert ds.attrs[str(names.product_name)] == \
        str(dim['Dataset_Id']['DATASET_NAME'])
    # The reader stores the parent of the resolved product root.
    assert ds.attrs[str(names.input_directory)] == \
        str(product_dir.resolve().parent)

    if variant == 'DE-1':
        expected_dt = f'{scene["IMAGING_DATE"]}T{scene["IMAGING_TIME"]}'
        expected_res = float(_num(scene['THEORETICAL_RESOLUTION']))
        assert expected_res == 22.0
    else:
        # DE-2 keeps the full START_TIME (v1 produced a date-only value).
        expected_dt = str(xml_value(scene['START_TIME']))
        expected_res = float(_num(scene['PIXEL_RESOLUTION_X']))
        assert expected_res == 3.0
    assert ds.attrs[str(names.datetime)] == expected_dt
    assert ds.attrs[str(names.resolution)] == expected_res

    # CRS straight from the GeoTIFF (DE-1 has none: identity + GCPs).
    with rasterio.open(str(_tif(product_dir))) as src:
        expected_crs = str(src.crs) if src.crs else 'EPSG:4326'
    assert ds.attrs[str(names.crs)] == expected_crs
    assert {'DE-1': 'EPSG:4326', 'DE-2': 'EPSG:32631'}[variant] == \
        ds.attrs[str(names.crs)]

    assert ds.attrs['_srf_getter'] == 'eoread.readers.geosat.read_geosat_srf'
    assert ds.attrs['_srf_getter_arg'] == variant

    assert ds[str(names.cwav)].dtype == 'float64'
    np.testing.assert_array_equal(ds[str(names.cwav)].values, expected_cwav)


def test_v1_compat(ds):
    old = v1(ds)
    assert 'Ltoa' in old
    assert 'schema_version' not in old.attrs


def test_angles_ground_truth(ds, dim):
    """The angles are constant over the scene: exact float32 values.

    Expected values mirror the reader's DIMAP lookups (Scene_Source leaves,
    some wrapped in {attributes, values} dicts).
    """
    scene = dim['Dataset_Sources']['Source_Information']['Scene_Source']
    expected = {
        'sza': np.float32(90.0 - float(xml_value(scene['SUN_ELEVATION']))),
        'saa': np.float32(float(xml_value(scene['SUN_AZIMUTH']))),
        'vza': np.float32(float(_num(scene['INCIDENCE_ANGLE']))),
        'vaa': np.float32(0.0),
    }
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for var, exp in expected.items():
        got = ds[var].isel(y=[0, ny // 2, ny - 1],
                           x=[0, nx // 2, nx - 1]).compute().values
        assert np.array_equal(got, np.full(got.shape, exp)), var


def test_latlon_gcp(product_dir, ds):
    """DE-1: the spline reproduces every GCP at its half-pixel center.

    The GCP rows/cols are half-integers (pixel centers); v2 row r carries
    file row ny - 1 - r.
    """
    if _variant(ds) != 'DE-1':
        pytest.skip('GCP geolocation is a DE-1 feature')
    with rasterio.open(str(_tif(product_dir))) as src:
        gcps, _ = src.gcps
    assert len(gcps) == 25, f'expected the 5x5 GCP grid, got {len(gcps)}'

    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for g in gcps:
        r_idx = int(g.row + 0.5)
        c_idx = int(g.col + 0.5)
        assert abs(g.row + 0.5 - r_idx) < 1e-9, g.row
        assert abs(g.col + 0.5 - c_idx) < 1e-9, g.col
        # rasterio GCP.x is longitude, GCP.y latitude.
        lat = ds['latitude'].isel(y=ny - 1 - r_idx, x=c_idx).values
        lon = ds['longitude'].isel(y=ny - 1 - r_idx, x=c_idx).values
        assert abs(lat - g.y) < 1e-5, (g.row, g.col)
        assert abs(lon - g.x) < 1e-5, (g.row, g.col)

    # Corner pixels sit one pixel outside the GCP grid; the cubic
    # extrapolation must stay inside the GCP bounding box.
    lats = np.array([g.y for g in gcps])
    lons = np.array([g.x for g in gcps])
    for r, c in ((0, 0), (ny - 1, 0), (0, nx - 1), (ny - 1, nx - 1)):
        lat = ds['latitude'].isel(y=r, x=c).values
        lon = ds['longitude'].isel(y=r, x=c).values
        assert lats.min() - 1e-3 <= lat <= lats.max() + 1e-3, (r, c)
        assert lons.min() - 1e-3 <= lon <= lons.max() + 1e-3, (r, c)


def test_latlon_utm(product_dir, ds):
    """DE-2: independent UTM pixel-center transform via pyproj."""
    if _variant(ds) != 'DE-2':
        pytest.skip('affine UTM geolocation is a DE-2 feature')
    with rasterio.open(str(_tif(product_dir))) as src:
        crs, t = str(src.crs), src.transform

    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    proj = Proj(crs)
    for r, c in ((0, 0), WIN1, (ny // 2, nx // 2), (ny - 1, nx - 1)):
        # Pixel centers in the affine grid (the latlon_utm formula).
        X = t.c + t.a * (c + 0.5) + t.b * (r + 0.5)
        Y = t.f + t.d * (c + 0.5) + t.e * (r + 0.5)
        lon_e, lat_e = proj(X, Y, inverse=True)
        lat = ds['latitude'].isel(y=ny - 1 - r, x=c).values
        lon = ds['longitude'].isel(y=ny - 1 - r, x=c).values
        assert abs(lat - lat_e) < 1e-5, (r, c)
        assert abs(lon - lon_e) < 1e-5, (r, c)


def test_nodata_mask(product_dir, ds, dim):
    """v2 NaN positions must equal the raw DN nodata flag on the edge lines.

    DE-1 carries no nodata flag: the whole product must be unmasked.
    """
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    lines = [
        {str(names.rows): 0},
        {str(names.rows): ny - 1},
        {str(names.columns): 0},
        {str(names.columns): nx - 1},
    ]
    if _variant(ds) != 'DE-2':
        for i, label in enumerate(ds.coords[str(names.bands)].values):
            for sel in lines:
                v2 = ds['Ltoa'].isel(bands=i, **sel).compute().values
                assert not np.isnan(v2).any(), (label, sel)
        return

    mapping = _raw_band_map(product_dir, dim)
    for i, label in enumerate(ds.coords[str(names.bands)].values):
        raw, j = mapping[label]
        for sel in lines:
            v2 = ds['Ltoa'].isel(bands=i, **sel).compute().values
            dn = raw.isel(bands=j, **sel).values
            assert np.array_equal(np.isnan(v2), (dn == 0.0)), (label, sel)


def test_exit_probe(product_dir, tmp_path):
    """A reader held until process exit must not emit teardown noise.

    rioxarray leaves one rasterio dataset open per lazily opened file; the
    atexit close (install_exit_cleanup) must release them before the
    interpreter finalizes, otherwise CPython >= 3.12 prints a nested
    'Error in sys.excepthook' pair for each of them.
    """
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        "from eoread.readers.geosat import Level1_GEOSAT\n"
        f"ds = Level1_GEOSAT({str(product_dir)!r})\n"
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
        Level1_GEOSAT(Path('/nonexistent/geosat_product'))
