#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the Pleiades-Neo reader (PMS mode).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

Documented deviations from the v1 implementation (see REFACTORING.md):

- south-up orientation: v2 row i carries file row ny - 1 - i.
- all six spectral bands are kept (sorted by central wavelength); the v1
  reader silently dropped the red-edge and deep-blue bands.
- nodata pixels (DN == 0 per the per-file Special_Value tables) become NaN;
  a plain rioxarray open keeps them as zero data (masked=False default).
- payload, angles and lat/lon are float64 end to end, as in v1.

The product stores unsigned view/solar azimuths in [0, 360) while the naming
registry declares [0, 180], so `generic.Test.angles` does not apply to this
sample (view azimuth ~194.5 deg); test_angles replaces it with scene-aware
ranges (OLCI/MSI/MERIS/PACE/SGLI precedent).
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rioxarray
from core.geo.naming import names
from core.table import read_xml

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.readers.pleiades import _DimapBase
from eoread.readers.pleiades_neo import Level1_PNEO
from eoread.tools import _num, xml_value

# Fixed check window in north-up (file) rows/columns; well inside the
# 2819 x 2780 strip.
WIN1 = (100, 200)


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so reader datasets are released promptly.
    yield
    gc.collect()


@pytest.fixture(scope='session')
def product_dir():
    return get_sample(Level1_PNEO)


@pytest.fixture(scope='session')
def strip_dim(product_dir):
    strip, dim_path = Level1_PNEO._find_strip(product_dir)
    return strip, read_xml(str(dim_path))


@pytest.fixture(scope='session')
def ds(product_dir):
    return Level1_PNEO(product_dir, chunks=500)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _bilinear(verts, r, c, ny):
    """Independent bilinear lat/lon at v2 pixel (r, c) from the DIMAP corners.

    The DIMAP vertices are in file space (row 0 = north); v2 row r carries
    file row ny - 1 - r, which is where the corner grid is evaluated. The
    plain bilinear form is evaluated in float64 with targets clipped to the
    corner span, as the reader does; the two must agree within float64
    rounding.
    """
    tl, tr, bl, br = sorted(verts, key=lambda v: (int(v['ROW']), int(v['COL'])))
    y0, y1 = float(int(tl['ROW']) - 1), float(int(bl['ROW']) - 1)
    x0, x1 = float(int(tl['COL']) - 1), float(int(tr['COL']) - 1)
    fy = min(max(float(ny - 1 - r), y0), y1)  # v2 row -> file row
    fx = min(max(float(c), x0), x1)
    vy = (fy - y0) / (y1 - y0)
    vx = (fx - x0) / (x1 - x0)
    out = []
    for key in ('LAT', 'LON'):
        g = np.array([[float(tl[key]), float(tr[key])],
                      [float(bl[key]), float(br[key])]], dtype='float64')
        top = (1.0 - vx) * g[0, 0] + vx * g[0, 1]
        bot = (1.0 - vx) * g[1, 0] + vx * g[1, 1]
        out.append((1.0 - vy) * top + vy * bot)
    return out


def _data_files(dim):
    files = dim['Raster_Data']['Data_Access']['Data_Files']
    if isinstance(files, dict):
        files = [files]
    return files


def _raw_band_map(strip, dim):
    """{band label: (south-up raw DN DataArray, in-file band index)}."""
    mapping = {}
    for entry in _data_files(dim):
        href = str(entry['Data_File']['DATA_FILE_PATH']['attributes']['href'])
        indices = entry['Raster_Display']['Raster_Index_List']['Raster_Index']
        if isinstance(indices, dict):
            indices = [indices]
        raw = rioxarray.open_rasterio(str(strip / href), masked=False,
                                      chunks=500)
        raw = raw.isel(y=slice(None, None, -1)).rename(band='bands')  # north-up flip
        for r in indices:
            mapping[str(r['BAND_ID'])] = (raw, int(r['BAND_INDEX']) - 1)
    return mapping


def _expected_bands_cwav(dim):
    """(band order, cwav array) recomputed from the DIMAP band table."""
    bml = (dim['Radiometric_Data']['Radiometric_Calibration']
           ['Instrument_Calibration']['Band_Measurement_List'])
    spec = {s['BAND_ID']: s for s in bml['Band_Spectral_Range']}
    cwav = {b: (float(_num(s['FWHM']['MIN'])) + float(_num(s['FWHM']['MAX']))) / 2
            for b, s in spec.items()}
    order = sorted(cwav, key=lambda b: cwav[b])
    return order, np.array([cwav[b] for b in order], dtype='float64')


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_instantiation(product_dir):
    # default chunks
    Level1_PNEO(product_dir)


def test_main(ds):
    generic.Test.main(ds, angle_data=True)


def test_conformance(ds):
    generic.Test.v2_conformance(ds, primary='Rtoa')


def test_latlon(ds):
    # Tilted strip: lat varies along the western edge by more than 1e-7 deg
    generic.Test.latlon(ds, flat_proj=False)


def test_angles(ds):
    """Angle ranges; deliberately not `generic.Test.angles`.

    The product stores unsigned view/solar azimuths in [0, 360) while the
    naming registry declares [0, 180] — a documented registry inconsistency
    (OLCI/MSI/MERIS/PACE/SGLI precedent). The pinned sample's view azimuth is
    ~194.5 deg, which the registry check rejects. The zenith angles are
    absolute and stay within [0, 90].
    """
    sel = {str(names.rows): slice(None, None, 50),
           str(names.columns): slice(None, None, 50)}
    for var, lo, hi in ((str(names.sza), 0., 90.), (str(names.vza), 0., 90.),
                        (str(names.saa), 0., 360.), (str(names.vaa), 0., 360.)):
        v = ds[var].isel(sel).values
        assert v.min() >= lo - 1e-3 and v.max() <= hi + 1e-3, \
            f'{var} out of range [{lo}, {hi}], got [{v.min()}, {v.max()}]'

    # Scene facts (pinned sample): view azimuth ~194.5 deg — the >180 deg
    # value that the registry-based check rejects; solar azimuth ~164.4 deg.
    vaa = ds[str(names.vaa)].isel(sel).values
    assert 190. < vaa.min() and vaa.max() < 200., \
        f'view azimuth out of scene band: [{vaa.min()}, {vaa.max()}]'


def test_lazy_load(ds):
    generic.Test.lazy_load(ds)


def test_subset(product_dir):
    ds = Level1_PNEO(product_dir, metadata_template=[])
    generic.Test.subset(ds)


def test_plot(request, ds):
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    generic.plot(request, ds, 'NIR',
                 poi={str(names.rows): ny // 2, str(names.columns): nx // 2})


def test_bands_attrs(product_dir, ds, strip_dim):
    _, dim = strip_dim
    src = _DimapBase._strip_source(dim)

    order, cwav_expected = _expected_bands_cwav(dim)
    assert list(ds.coords[str(names.bands)].values) == order
    # All six bands are kept; the v1 reader dropped RE and DB.
    assert set(order) == {'DB', 'B', 'G', 'R', 'RE', 'NIR'}
    assert list(ds.coords[str(names.bgroup)].values) == ['bands_vnir'] * 6
    assert ds.attrs[str(names.crs)] == 'EPSG:4326'
    assert ds.attrs[str(names.sensor)] == 'PNEO'
    assert ds.attrs[str(names.platform)] == f'{src["MISSION"]}-{src["MISSION_INDEX"]}'
    assert ds.attrs[str(names.shortname)] == str(src['BAND_MODE'])
    assert ds.attrs[str(names.product_name)] == str(xml_value(
        dim['Dataset_Identification']['DATASET_NAME']))
    assert ds.attrs[str(names.datetime)] == \
        f'{src["IMAGING_DATE"]}T{src["IMAGING_TIME"]}'.removesuffix('Z')
    # The reader stores the parent of the resolved product root.
    assert ds.attrs[str(names.input_directory)] == \
        str(product_dir.resolve().parent)

    gsd = dim['Geometric_Data']['Use_Area']['Located_Geometric_Values'][1] \
        ['Ground_Sample_Distance']
    expected_res = (float(xml_value(gsd['GSD_ACROSS_TRACK']))
                    + float(xml_value(gsd['GSD_ALONG_TRACK']))) / 2.0
    assert ds.attrs[str(names.resolution)] == pytest.approx(expected_res)

    # The SRF getter is inherited from the Pleiades family base module.
    assert ds.attrs['_srf_getter'] == 'eoread.readers.pleiades.read_srf_pleiades'
    assert ds.attrs['_srf_getter_arg'] == f'{src["MISSION"]}{src["MISSION_INDEX"]}'

    # Central wavelengths: FWHM bounds are already in nm.
    assert ds[str(names.cwav)].dtype == 'float64'
    np.testing.assert_array_equal(ds[str(names.cwav)].values, cwav_expected)
    # The four bands shared with v1 carry the pinned central wavelengths.
    np.testing.assert_array_equal(
        ds[str(names.cwav)].sel(bands=['B', 'G', 'R', 'NIR']).values,
        [483., 562., 654.5, 828.])


def test_v1_compat(ds):
    old = v1(ds)
    assert 'Rtoa' in old
    assert 'schema_version' not in old.attrs


def test_angles_ground_truth(ds, strip_dim):
    """At the three Use_Area profile rows the interpolation hits its knots.

    v2 row r carries file row ny - 1 - r; the Top / Center / Bottom entries
    sit at file rows 0, (ny-1)/2, ny-1, so the expected values are the raw
    DIMAP numbers cast to float64 (bit-exact).
    """
    _, dim = strip_dim
    ny = ds.sizes[str(names.rows)]
    vals = dim['Geometric_Data']['Use_Area']['Located_Geometric_Values']

    # (v2 row, Use_Area entry index): south-up flips the profile.
    for r, idx in ((ny - 1, 0), (ny // 2, 1), (0, 2)):
        e = vals[idx]
        expected = {
            'sza': np.float64(90.0 - float(xml_value(e['Solar_Incidences']['SUN_ELEVATION']))),
            'saa': np.float64(float(xml_value(e['Solar_Incidences']['SUN_AZIMUTH']))),
            'vza': np.float64(float(xml_value(e['Acquisition_Angles']['VIEWING_ANGLE']))),
            'vaa': np.float64(float(xml_value(e['Acquisition_Angles']['AZIMUTH_ANGLE']))),
        }
        for var, exp in expected.items():
            got = ds[var].isel(y=r, x=[0, 100, -1]).compute().values
            # np.array_equal does not broadcast: compare against a full array.
            assert np.array_equal(got, np.full(got.shape, exp)), (var, r)


def test_latlon_bilinear(ds, strip_dim):
    """Independent bilinear evaluation from the four DIMAP footprint corners."""
    _, dim = strip_dim
    verts = dim['Dataset_Content']['Dataset_Extent']['Vertex']
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for r, c in ((ny // 2, nx // 2), (ny // 3, 2 * nx // 3), WIN1):
        lat_e, lon_e = _bilinear(verts, r, c, ny)
        assert abs(ds['latitude'].isel(y=r, x=c).values - lat_e) < 2e-5, (r, c)
        assert abs(ds['longitude'].isel(y=r, x=c).values - lon_e) < 2e-5, (r, c)

    # The sample's footprint vertices sit exactly on the raster corners
    # (ROW 1..ny, COL 1..nx), so the v2 south-west / north-east pixels carry
    # the BL / TR vertex values essentially exactly (float64 end to end).
    tl, tr, bl, br = sorted(verts, key=lambda v: (int(v['ROW']), int(v['COL'])))
    assert abs(ds['latitude'].isel(y=0, x=0).values - float(bl['LAT'])) < 1e-5
    assert abs(ds['longitude'].isel(y=0, x=0).values - float(bl['LON'])) < 1e-5
    assert abs(ds['latitude'].isel(y=ny - 1, x=nx - 1).values - float(tr['LAT'])) < 1e-5
    assert abs(ds['longitude'].isel(y=ny - 1, x=nx - 1).values - float(tr['LON'])) < 1e-5


def test_nodata_mask(ds, strip_dim):
    """v2 NaN positions must equal the raw DN nodata flag on the edge lines.

    The MS product spans two JP2 files with their own Special_Value tables;
    each v2 band is checked against its source file and in-file band index.
    """
    strip, dim = strip_dim
    values = set()
    for entry in _data_files(dim):
        values.update(_DimapBase._nodata_values(entry['Raster_Display']))
    assert values == {0.0}, f'expected DN==0 nodata flags, got {values}'

    mapping = _raw_band_map(strip, dim)
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    lines = [
        {str(names.rows): 0},
        {str(names.rows): ny - 1},
        {str(names.columns): 0},
        {str(names.columns): nx - 1},
    ]
    for i, label in enumerate(ds.coords[str(names.bands)].values):
        raw, j = mapping[label]
        for sel in lines:
            v2 = ds['Rtoa'].isel(bands=i, **sel).compute().values
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
        "from eoread.readers.pleiades_neo import Level1_PNEO\n"
        f"ds = Level1_PNEO({str(product_dir)!r})\n"
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
        Level1_PNEO(Path('/nonexistent/pneo_product'))
