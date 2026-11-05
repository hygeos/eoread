#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the MERIS readers (L1).

Two product families are covered:

- N1 products (epr) read by `Level1_MERIS`;
- NASA-repackaged NetCDF SAFE (`.SEN3`) products read by
  `Level1_MERIS_SAFE` (see its docstring in eoread/readers/meris.py). The
  SAFE section pins the xarray engine to 'h5netcdf' for the whole test
  process (dual-libhdf5, see tests/test_olci.py).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

Documented deviations from the v1 implementation (see REFACTORING.md):

- south-up orientation: v2 row i carries file row ny - 1 - i.
- ``F0`` / ``wav`` are always emitted; the v1 reader gated them behind
  ``read_auxdata=False``.
- ``resolution`` comes from the SPH COLUMN_SPACING (1040 m on this RR sample);
  v1 hardcoded 300 m for both variants.
- Azimuth angles keep the product's absolute [0, 360) convention; the naming
  registry declares [0, 180] — a documented registry inconsistency (OLCI/MSI).
"""

import gc
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from core.geo.naming import names
from core.table import read_xml

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.flags import FlagsInit, GenericFlags
from eoread.reader import ProductError, UnsupportedProductType
from eoread.readers.meris import (Level1_MERIS, Level1_MERIS_SAFE,
                                  _AUXDATA_DIR, _read_date)

# Fixed check window: north-up raster rows x columns, well inside the
# 15009 x 1121 scene.
WIN_ROWS = np.array([4000, 7500, 11000])
WIN_COLS = np.array([300, 560, 820])


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so reader datasets are released promptly.
    yield
    gc.collect()


@pytest.fixture(scope='session')
def l1_dir():
    return get_sample(Level1_MERIS)


@pytest.fixture(scope='session')
def l1(l1_dir):
    return Level1_MERIS(l1_dir, chunks=500)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------

def test_l1_instantiation(l1_dir, chunks):
    Level1_MERIS(l1_dir, chunks=chunks)


def test_l1_main(l1):
    generic.Test.main(l1, angle_data=True)


def test_l1_conformance(l1):
    generic.Test.v2_conformance(l1, primary='Ltoa')


def test_l1_latlon(l1):
    """Geolocation checks against the SPH corner grid.

    Deliberately not `generic.Test.latlon`: this scene's ground track curves
    strongly (the across-track longitude orientation flips between the two
    ends), and it crosses the polar region — latitude along column 0 rises to
    ~86.7N and curls back down around the pole — so a strict monotonicity
    check would raise on valid data.
    """
    ny, nx = l1.sizes[str(names.rows)], l1.sizes[str(names.columns)]

    # Latitude along column 0: at most one local maximum (the polar wiggle),
    # never a local minimum — a single ground-track segment cannot turn back
    # on itself more than once.
    lat_c0 = l1['latitude'].isel(x=0).compute().values
    s = np.sign(np.diff(lat_c0))
    s = s[s != 0]  # ignore exact zeros (float32 rounding at the wiggle peak)
    turns_down = int(((s[:-1] > 0) & (s[1:] < 0)).sum())
    turns_up = int(((s[:-1] < 0) & (s[1:] > 0)).sum())
    assert turns_up == 0 and turns_down <= 1, \
        f'latitude along column 0 turns back {turns_up}x up / {turns_down}x down'

    # Corner pixels against the SPH nominal corners (stored in 1e-6 deg). The
    # EPR corner naming is crosswise: pixel (row 0, col 0) is FIRST_LAST.
    md = l1.attrs['metadata']
    corners = (((0, 0), 'LAST_LAST'), ((0, nx - 1), 'LAST_FIRST'),
               ((ny - 1, 0), 'FIRST_LAST'), ((ny - 1, nx - 1), 'FIRST_FIRST'))
    for (r, c), key in corners:
        lat = l1['latitude'].isel(y=r, x=c).compute().values
        lon = l1['longitude'].isel(y=r, x=c).compute().values
        assert abs(lat - md[f'{key}_LAT'] / 1e6) < 1e-4, (key, 'lat', lat)
        assert abs(lon - md[f'{key}_LONG'] / 1e6) < 1e-4, (key, 'lon', lon)


def test_l1_angles(l1):
    """Sensor-native angle ranges; deliberately not `generic.Test.angles`.

    The registry declares [0, 180] for every azimuth, but MERIS stores
    absolute geographic azimuths in [0, 360) — a documented registry
    inconsistency (see the OLCI/MSI entries in REFACTORING.md). Folding would
    be geometrically wrong for absolute azimuths, so v2 keeps the raw values.
    """
    ranges = {str(names.sza): (0., 90.), str(names.vza): (0., 90.),
              str(names.saa): (0., 360.), str(names.vaa): (0., 360.)}
    for name, (lo, hi) in ranges.items():
        v = l1[name].isel(y=slice(None, None, 50),
                          x=slice(None, None, 25)).compute().values
        assert v.min() >= lo - 1e-3 and v.max() <= hi + 1e-3, \
            f'{name} out of range [{lo}, {hi}], got [{v.min()}, {v.max()}]'


def test_l1_lazy_load(l1):
    generic.Test.lazy_load(l1)


def test_l1_flagreader(l1):
    generic.Test.flagreader(l1)


def test_l1_flag_values(l1):
    """L1_INVALID must reproduce `l1_flags != 0` on a 100x100 slice."""
    sub = l1.isel(y=slice(500, 600), x=slice(300, 400))
    flags = FlagsInit({GenericFlags.L1_INVALID: 4}, 'uint8',
                      l1.attrs['_flag_reader']).map_blocks(sub)
    f = flags['flags'].compute()
    assert np.array_equal((f & 4).astype(bool), (sub['l1_flags'] != 0).compute().values)


def test_l1_subset(l1_dir):
    ds = Level1_MERIS(l1_dir, metadata_template=[])
    generic.Test.subset(ds)


def test_l1_plot(request, l1):
    generic.plot(request, l1, 'B4')


# --------------------------------------------------------------------------
# Band table and attributes
# --------------------------------------------------------------------------

def test_l1_bands_cwav(l1):
    nb = 15
    # Official MERIS band names: B1..B15.
    assert list(l1.coords[str(names.bands)].values) == [f'B{i + 1}' for i in range(nb)]
    assert list(l1.coords[str(names.bgroup)].values) == ['bands_vnir'] * nb
    cwav = l1[str(names.cwav)].values
    assert cwav.dtype == np.float64
    want = np.asarray(l1.attrs['metadata']['BAND_WAVELEN'], dtype='uint32') / 1e3
    assert np.array_equal(cwav, want)


def test_l1_attrs(l1, l1_dir):
    a = l1.attrs
    md = a['metadata']
    assert a[str(names.sensor)] == 'MERIS'
    assert a[str(names.platform)] == 'ENVISAT'
    assert a[str(names.crs)] == 'EPSG:4326'

    product = md['PRODUCT']
    assert isinstance(product, str)  # single-element bytes fields are decoded
    assert a[str(names.product_name)] == product
    assert a[str(names.shortname)] == re.split(r'\d{8}', product)[0]

    # Data-driven resolution (documented deviation from v1's hardcoded 300)
    assert a[str(names.resolution)] == int(md['COLUMN_SPACING'])

    dstart = _read_date(md['SENSING_START'])
    dstop = _read_date(md['SENSING_STOP'])
    assert a[str(names.datetime)] == (dstart + (dstop - dstart) // 2).isoformat()

    assert a[str(names.input_directory)] == str(Path(l1_dir).parent)
    assert a['user_guide'].startswith('https://archive.org/')
    assert a['_flag_reader'] == 'eoread.readers.meris.Level1_MERIS'
    assert a['_srf_getter'] == 'eotools.srf.get_SRF_meris'
    assert a['_srf_getter_arg'] == ''


def test_l1_v1_compat(l1):
    old = v1(l1)
    # identity + schema_version strip; F0/wav remain (documented superset —
    # the v1 default output did not carry them)
    assert 'Ltoa' in old and 'F0' in old and str(names.wav) in old
    assert 'schema_version' not in old.attrs


# --------------------------------------------------------------------------
# F0 / wav (always emitted in v2)
# --------------------------------------------------------------------------

def test_l1_F0_wav(l1):
    """Ground-truth check of the per-pixel table gather, incl. NaN masking."""
    ny = l1.sizes[str(names.rows)]
    r = WIN_ROWS.tolist()
    c = WIN_COLS.tolist()
    j = (ny - 1 - np.array(r)).tolist()

    for var, table_name, dtype in ((str(names.F0), 'sun_spectral_flux', 'float32'),
                                   (str(names.wav), 'central_wavelen', 'float64')):
        # Leading detector column dropped (same convention as the reader)
        table = np.loadtxt(_AUXDATA_DIR / f'{table_name}_rr.txt',
                           delimiter='\t', skiprows=1, dtype=dtype)[:, 1:]
        det = l1['detector_index'].isel(y=j, x=c).compute().values
        valid = det >= 0
        want = table[np.clip(det, 0, None)].transpose(2, 0, 1)
        want = np.where(valid[None, :, :], want, np.nan)
        got = l1[var].isel(y=j, x=c).compute().values
        assert got.dtype == np.dtype(dtype), var
        assert np.array_equal(got, want, equal_nan=True), var

    # The sample carries invalid pixels (detector_index < 0): F0/wav must be
    # NaN there.
    det_full = l1['detector_index'].compute().values
    invalid = np.argwhere(det_full < 0)
    assert invalid.size > 0, 'expected at least one invalid pixel in the sample'
    r0, c0 = invalid[0]
    for var in (str(names.F0), str(names.wav)):
        assert np.isnan(l1[var].isel(y=int(r0), x=int(c0)).compute().values).all(), var


# --------------------------------------------------------------------------
# Process hygiene / guards
# --------------------------------------------------------------------------

def test_l1_exit_probe(l1_dir, tmp_path):
    """A reader held until process exit must not emit teardown noise.

    epr.Product is never explicitly closed; the interpreter exits cleanly
    with it held open (no atexit cleanup is needed — pinned here).
    """
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        "from eoread.readers.meris import Level1_MERIS\n"
        f"ds = Level1_MERIS({str(l1_dir)!r})\n"
        "x = ds['Ltoa'].isel(y=slice(100, 200), x=slice(100, 200)).compute()\n"
        "print('COMPUTED', float(x.mean()))\n",
    )
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'COMPUTED' in proc.stdout
    assert 'excepthook' not in proc.stderr, \
        'teardown noise at exit: the open epr product emitted shutdown errors'


def test_variant_detection():
    """RR/FR are detected from the decoded PRODUCT prefix (loud raise otherwise)."""
    probe = object.__new__(Level1_MERIS)  # detect_variant only needs VARIANTS
    assert probe.detect_variant({'PRODUCT': 'MER_RR__1PRACR20080701_014028_N1'}) == 'rr'
    assert probe.detect_variant({'PRODUCT': 'MER_FR_1PNPDE20060822_092058_N1'}) == 'fr'
    with pytest.raises(UnsupportedProductType):
        probe.detect_variant({'PRODUCT': 'MER_L2__1PRACR20080701_N1'})


def test_missing_path():
    with pytest.raises(AssertionError):
        Level1_MERIS(Path('/nonexistent/meris_product'))


# ===========================================================================
# Level1_MERIS_SAFE (NASA-repackaged NetCDF SAFE / .SEN3 products)
# ===========================================================================

_ENGINE = 'h5netcdf'


def _readable(path):
    """Skip cleanly if the staged sample cannot be traversed/read.

    A .SEN3 granule is a directory tree: a missing execute bit at any level
    surfaces as PermissionError on the first open instead of a useful test
    failure.
    """
    if not os.access(path, os.R_OK | os.X_OK):
        pytest.skip(f'sample not readable (check permissions): {path}')
    return path


def _safe_band_descriptions(path):
    """The manifest `bandDescriptions` mapping (independent of reader parsing)."""
    manifest = read_xml(Path(path) / 'xfdumanifest.xml')
    info = next(o for o in manifest['metadataSection']['metadataObject']
                if o['attributes']['ID'] == 'merisProductInformation')
    return info['metadataWrap']['xmlData']['merisProductInformation'] \
        ['bandDescriptions']


@pytest.fixture(scope='session')
def safe_rr_dir():
    return _readable(get_sample(Level1_MERIS_SAFE, 'MERIS_L1_SAFE_RRG'))


@pytest.fixture(scope='session')
def safe_fr_dir():
    return _readable(get_sample(Level1_MERIS_SAFE, 'MERIS_L1_SAFE_FRG'))


@pytest.fixture
def safe_rr(safe_rr_dir):
    return Level1_MERIS_SAFE(safe_rr_dir, chunks=500, engine=_ENGINE)


@pytest.fixture
def safe_fr(safe_fr_dir):
    return Level1_MERIS_SAFE(safe_fr_dir, chunks=500, engine=_ENGINE)


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------

def test_safe_instantiation(safe_rr_dir, chunks):
    Level1_MERIS_SAFE(safe_rr_dir, chunks=chunks, engine=_ENGINE)


def test_safe_main(safe_rr):
    generic.Test.main(safe_rr, angle_data=True)


def test_safe_conformance(safe_rr):
    generic.Test.v2_conformance(safe_rr, primary='Ltoa')


def test_safe_latlon(safe_rr):
    """NaN-aware orientation check.

    The SAFE geolocation is NaN outside the swath (fully invalid edge rows,
    partially filled first/last rows), so a raw first/last pixel comparison
    would compare fill values; use the first/last *valid* pixel instead.
    Latitude is not asserted monotonic along-track: this granule's ground
    track turns near the South Pole (ENVISAT inclination ~98.5 deg).
    """
    rows, cols = str(names.rows), str(names.columns)
    ny, nx = safe_rr.sizes[rows], safe_rr.sizes[cols]

    lat = safe_rr[str(names.lat)].isel({cols: nx // 2}).values
    valid = np.isfinite(lat)
    assert valid.sum() >= 2
    assert lat[valid][0] < lat[valid][-1], 'latitude must be south-up'
    assert -90. <= lat[valid].min() and lat[valid].max() <= 90.

    lon = safe_rr[str(names.lon)].isel({rows: ny // 2}).values
    vlon = np.isfinite(lon)
    assert vlon.sum() >= 2
    assert lon[vlon][0] < lon[vlon][-1], 'longitude must be west-to-east'

    # The orientation flips re-index the grid to a plain arange.
    assert np.array_equal(safe_rr[rows].values, np.arange(ny))
    assert np.array_equal(safe_rr[cols].values, np.arange(nx))


def _safe_check_angles(ds, expect_neg_saa):
    """Signed-azimuth range check on a decimated sample (never the full grid).

    Deliberately not `generic.Test.angles`: that checks against the naming
    registry minv/maxv, which declare [0, 180] for every azimuth, while the
    SAFE products store SAA/VAA as signed degrees in [-180, 180] — a
    documented registry inconsistency (REFACTORING.md).
    """
    ranges = {str(names.sza): (0., 90.), str(names.vza): (0., 90.),
              str(names.saa): (-180., 180.), str(names.vaa): (-180., 180.)}
    got = {}
    for name, (lo, hi) in ranges.items():
        v = ds[name].isel(y=slice(None, None, 50), x=slice(None, None, 25)).values
        assert np.isfinite(v).all(), f'{name}: NaNs on the full grid'
        assert v.min() >= lo - 1e-3 and v.max() <= hi + 1e-3, \
            f'{name} out of range [{lo}, {hi}]: got [{v.min()}, {v.max()}]'
        got[name] = v
    saa, vaa = got[str(names.saa)], got[str(names.vaa)]
    assert (vaa < 0).any(), 'expected negative viewing azimuths'
    if expect_neg_saa:
        assert (saa < 0).any() and (saa > 0).any(), \
            'sun azimuth expected to span both hemispheres'
    else:
        assert (saa > 0).all(), 'sun azimuth stays positive on this sample'


def test_safe_angles_rr(safe_rr):
    # The RRG sample's sun azimuth spans the full signed range.
    _safe_check_angles(safe_rr, expect_neg_saa=True)


def test_safe_angles_fr(safe_fr):
    # The FRG sample's sun azimuth stays in [127 deg, 167 deg]: no negative SAA.
    _safe_check_angles(safe_fr, expect_neg_saa=False)


# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------

def test_safe_lazy_load(safe_rr):
    generic.Test.lazy_load(safe_rr)


def test_safe_flagreader(safe_rr):
    generic.Test.flagreader(safe_rr)


def test_safe_flag_values(safe_rr):
    """L1_INVALID/LAND must reproduce getflag() on a 100x100 slice."""
    sub = safe_rr.isel(y=slice(500, 600), x=slice(300, 400))
    flags = FlagsInit({GenericFlags.L1_INVALID: 4, GenericFlags.LAND: 2},
                      'uint8', safe_rr.attrs['_flag_reader']).map_blocks(sub)
    f = flags['flags'].compute()
    inv = Level1_MERIS_SAFE.getflag(sub, GenericFlags.L1_INVALID).values
    land = Level1_MERIS_SAFE.getflag(sub, GenericFlags.LAND).values
    assert np.array_equal((f & 4).astype(bool), inv)
    assert np.array_equal((f & 2).astype(bool), land)


# --------------------------------------------------------------------------
# I/O behaviour
# --------------------------------------------------------------------------

def test_safe_subset(safe_rr_dir):
    ds = Level1_MERIS_SAFE(safe_rr_dir, metadata_template=[], engine=_ENGINE)
    generic.Test.subset(ds)


def test_safe_plot(request, safe_rr):
    generic.plot(request, safe_rr, 'B4')


# --------------------------------------------------------------------------
# Bands / spectral tables
# --------------------------------------------------------------------------

def test_safe_bands_cwav(safe_rr, safe_rr_dir):
    nb = 15
    assert list(safe_rr.coords[str(names.bands)].values) == \
        [f'B{i + 1}' for i in range(nb)]
    assert list(safe_rr.coords[str(names.bgroup)].values) == ['bands_vnir'] * nb

    # Central wavelengths from the manifest (independent of reader parsing).
    desc = _safe_band_descriptions(safe_rr_dir)
    want = np.array([float(desc[k]['centralWavelength'])
                     for k in desc if k != 'attributes'])
    cwav = safe_rr[str(names.cwav)].values
    assert cwav.dtype == np.float64
    assert np.array_equal(cwav, want)
    assert (np.diff(cwav) > 0).all()

    # Per-pixel wavelength table: the mean over valid detectors of each band
    # must sit within the band, i.e. close to its central wavelength.
    for i in range(nb):
        tab = safe_rr['lambda0'].isel(bands=i).values
        m = float(tab[tab > 0].mean())
        assert abs(m - cwav[i]) < 20., (i, m, cwav[i])


def test_safe_F0_wav(safe_rr):
    """Ground-truth check of the per-pixel table gather, incl. NaN masking."""
    ny = safe_rr.sizes[str(names.rows)]
    nx = safe_rr.sizes[str(names.columns)]
    r = [ny // 2 - 5, ny // 2, ny // 2 + 5]
    c = [nx // 3, nx // 2, 2 * nx // 3]

    det = safe_rr['detector_index'].isel(y=r, x=c).values
    assert det.dtype == np.int16
    valid = det >= 0

    for var, table_name in ((str(names.F0), 'solar_flux'), (str(names.wav), 'lambda0')):
        tab = safe_rr[table_name].values  # raw -1 fills kept
        want = np.take(tab, np.clip(det, 0, None).astype('intp'), axis=1)
        want = np.where(valid[None, :, :], want, np.nan)
        got = safe_rr[var].isel(y=r, x=c).values
        assert got.dtype == np.float32, var
        assert np.array_equal(got, want, equal_nan=True), var

    # The sample carries invalid pixels (detector_index < 0): F0/wav must be
    # NaN there.
    det_full = safe_rr['detector_index'].values
    invalid = np.argwhere(det_full < 0)
    assert invalid.size > 0, 'expected at least one invalid pixel in the sample'
    r0, c0 = (int(v) for v in invalid[0])
    for var in (str(names.F0), str(names.wav)):
        assert np.isnan(safe_rr[var].isel(y=r0, x=c0).values).all(), var


# --------------------------------------------------------------------------
# Attributes / variants / guards
# --------------------------------------------------------------------------

def test_safe_attrs(safe_rr, safe_rr_dir):
    a = safe_rr.attrs
    assert a[str(names.sensor)] == 'MERIS'
    assert a[str(names.platform)] == 'ENVISAT'
    assert a[str(names.crs)] == 'EPSG:4326'

    path = Path(safe_rr_dir)
    assert a[str(names.product_name)] == path.name
    assert a[str(names.shortname)] == 'ME_1_RRG___'
    assert a[str(names.input_directory)] == str(path.parent)

    # Data-driven resolution (manifest acSpatialSampling), int-ified.
    assert a[str(names.resolution)] == 1040

    # datetime = midpoint of the manifest acquisition period.
    manifest = read_xml(path / 'xfdumanifest.xml')
    objs = {o['attributes']['ID']: o
            for o in manifest['metadataSection']['metadataObject']}
    acq = objs['acquisitionPeriod']['metadataWrap']['xmlData']['acquisitionPeriod']
    start = datetime.fromisoformat(acq['startTime'])
    stop = datetime.fromisoformat(acq['stopTime'])
    assert a[str(names.datetime)] == (start + (stop - start) / 2.).isoformat()

    assert a['user_guide'].startswith('https://archive.org/')
    assert a['_flag_reader'] == 'eoread.readers.meris.Level1_MERIS_SAFE'
    assert a['_srf_getter'] == 'eotools.srf.get_SRF_meris'
    assert a['_srf_getter_arg'] == ''


def test_safe_fr(safe_fr):
    """The FRG sample is detected, sized and resolved as full resolution."""
    a = safe_fr.attrs
    assert a[str(names.shortname)] == 'ME_1_FRG___'
    assert a[str(names.resolution)] == 260
    assert safe_fr.sizes[str(names.rows)] == 8193
    assert safe_fr.sizes[str(names.columns)] == 4481
    assert safe_fr.sizes['detectors'] == 3700


def test_safe_v1_compat(safe_rr):
    old = v1(safe_rr)
    # identity + schema_version strip (no documented break — this product
    # family has no legacy reader at all)
    assert 'Ltoa' in old and str(names.F0) in old and str(names.wav) in old
    assert 'schema_version' not in old.attrs
    assert 'schema_version' in safe_rr.attrs


def test_safe_variant_detection():
    """RR/FR are detected from the manifest productType (loud raise otherwise)."""
    probe = object.__new__(Level1_MERIS_SAFE)  # detect_variant only needs VARIANTS
    assert probe.detect_variant({'productType': 'ME_1_RRG___'}) == 'RR'
    assert probe.detect_variant({'productType': 'ME_1_FRG___'}) == 'FR'
    with pytest.raises(UnsupportedProductType):
        probe.detect_variant({'productType': 'ME_2_LRR____'})


def test_safe_missing_path():
    with pytest.raises(ProductError):
        Level1_MERIS_SAFE(Path('/nonexistent/meris_safe_product'), engine=_ENGINE)


def test_not_a_meris_safe_product(tmp_path):
    with pytest.raises(ProductError):
        Level1_MERIS_SAFE(tmp_path, engine=_ENGINE)
