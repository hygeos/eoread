#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the GCOM-C SGLI reader (L1B).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

The sample is a descending-track granule: the file stores lines in
acquisition order, so v1's output was north-up and v2 flips it — v2 row i
carries file row ny-1-i. The v1 geometry pipeline applied (1+Slope)*x instead
of Slope*x+Offset: lat/lon were doubled and the centidegree angles inflated
~101x. Geometry therefore pins against the product's raw tie data
(test_l1b_tie_values), Rtoa is recomputed from the file's raw counts
(test_l1b_rtoa_formula), and only the untransformed uint8/uint16 masks pin
bit-exactly.

The scene is a curved swath (latitude varies by several degrees along a
raster row), so `generic.Test.latlon`'s flat-projection check does not apply;
and the azimuth angles are signed relative values in [-180, 180] while the
naming registry declares [0, 180], so `generic.Test.angles` does not either.
Both are replaced by scene-aware checks below.
"""

import gc
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from core.geo.naming import names
from dateutil.parser import parse

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.flags import FlagsInit, GenericFlags
from eoread.reader import UnsupportedProductType
from eoread.readers.sgli import Level1B_SGLI, SGLI_CENTRAL_WAVELENGTHS

# Fixed check window: raster rows x columns, well inside the 7820x5000 scene.
WIN_ROWS = [1000, 4000, 7000]
WIN_COLS = [500, 2500, 4500]

# Tie-point fields: file name -> output name (v1's exact names).
TIE_FIELDS = (('Latitude', 'lat_tie'), ('Longitude', 'lon_tie'),
              ('Sensor_zenith', str(names.vza) + '_tie'),
              ('Sensor_azimuth', str(names.vaa) + '_tie'),
              ('Solar_zenith', str(names.sza) + '_tie'),
              ('Solar_azimuth', str(names.saa) + '_tie'))


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so reader datasets are released promptly.
    yield
    gc.collect()


@pytest.fixture(scope='session')
def l1_dir():
    return get_sample(Level1B_SGLI)


@pytest.fixture(scope='session')
def l1(l1_dir):
    return Level1B_SGLI(l1_dir, chunks=500)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------

def test_l1b_instantiation(l1_dir, chunks):
    Level1B_SGLI(l1_dir, chunks=chunks)


def test_l1b_main(l1):
    generic.Test.main(l1, angle_data=True)


def test_l1b_conformance(l1):
    generic.Test.v2_conformance(l1, primary='Rtoa')


def test_l1b_latlon(l1, l1_dir):
    """Geolocation checks against the product's own tie-point footprint.

    Deliberately not `generic.Test.latlon`: this is a curved swath, latitude
    varies by several degrees along a raster row, so the flat-projection
    invariance check would raise on valid data. Orientation is pinned here:
    v2 is south-up, col-0 latitude runs south to north (the sample granule is
    descending-track and stored north-up in the file; see the reader).
    """
    rows, cols = str(names.rows), str(names.columns)

    # Along-track orientation: at most one turn is allowed (polar orbits dip
    # back around the pole).
    prof = l1[str(names.lat)].isel({cols: 0}).values
    s = np.sign(np.diff(prof))
    nz = s[s != 0]
    n_max = int(((nz[:-1] > 0) & (nz[1:] < 0)).sum())
    n_min = int(((nz[:-1] < 0) & (nz[1:] > 0)).sum())
    assert prof[0] < prof[-1], 'col-0 latitude must run south to north'
    assert n_max == 0 and n_min <= 1, \
        f'unexpected turns in col-0 latitude ({n_max} maxima, {n_min} minima)'

    # Longitude runs west to east along the (south) edge row.
    row = l1[str(names.lon)].isel({rows: 0}).values
    assert row[0] < row[-1], 'row-0 longitude must run west to east'

    # Full-grid extent vs the raw tie-point values: linear interpolation stays
    # inside the tie convex hull, and the extreme ties sit on the grid edge,
    # so each full-grid bound is within half a pixel of the tie bound.
    tree = xr.open_datatree(l1_dir, engine='h5netcdf', phony_dims='sort')
    try:
        geo = tree['Geometry_data'].to_dataset()
        ties = {src: (np.float32(geo[src].attrs['Slope']) * geo[src].values
                      + np.float32(geo[src].attrs['Offset']))
                for src in ('Latitude', 'Longitude')}
    finally:
        tree.close()

    for var, tie in ((str(names.lat), ties['Latitude']),
                     (str(names.lon), ties['Longitude'])):
        v = l1[var].values
        assert v.min() >= tie.min() - 1e-6 and v.max() <= tie.max() + 1e-6, \
            f'{var} extent [{v.min()}, {v.max()}] outside tie hull ' \
            f'[{tie.min()}, {tie.max()}]'
        assert abs(v.min() - tie.min()) < 5e-2 and abs(v.max() - tie.max()) < 5e-2, \
            f'{var} extent [{v.min()}, {v.max()}] not on the tie hull edge ' \
            f'[{tie.min()}, {tie.max()}]'


def test_l1b_angles(l1):
    """Angle ranges; deliberately not `generic.Test.angles`.

    The product stores signed relative azimuths in [-180, 180] while the
    naming registry declares [0, 180] — a documented registry inconsistency
    (OLCI/MSI/MERIS/PACE precedent). The zenith angles are absolute and stay
    within [0, 90]. The scene facts pin the corrected magnitudes: v1's
    broken scale factor would inflate them ~101x.
    """
    sel = {str(names.rows): slice(None, None, 50),
           str(names.columns): slice(None, None, 50)}
    for var, lo, hi in ((str(names.sza), 0., 90.), (str(names.vza), 0., 90.),
                        (str(names.saa), -180., 180.), (str(names.vaa), -180., 180.)):
        v = l1[var].isel(sel).values
        assert v.min() >= lo - 1e-3 and v.max() <= hi + 1e-3, \
            f'{var} out of range [{lo}, {hi}], got [{v.min()}, {v.max()}]'

    # Scene facts (pinned sample): solar zenith spans ~24-47 deg across the
    # granule; sensor zenith peaks at ~50 deg off-nadir.
    vza = l1[str(names.vza)].isel(sel).values
    sza = l1[str(names.sza)].isel(sel).values
    assert vza.max() < 60., f'sensor zenith too large: {vza.max()}'
    assert 20. < sza.min() and sza.max() < 50., \
        f'solar zenith out of scene band: [{sza.min()}, {sza.max()}]'


# --------------------------------------------------------------------------
# Value pins against the product itself (v1 geometry was broken)
# --------------------------------------------------------------------------

def test_l1b_tie_values(l1, l1_dir):
    """Tie fields carry the full affine Slope*x + Offset.

    The v1 reader's (1+Slope)*x doubled lat/lon and inflated the centidegree
    angles ~101x; the product's raw tie data is the reference. The sample granule is
    north-up in file order, so v2's tie rows are mirrored — the test applies
    the same orientation rule as the reader (pinned independently by
    test_l1b_latlon).
    """
    tree = xr.open_datatree(l1_dir, engine='h5netcdf', phony_dims='sort')
    try:
        geo = tree['Geometry_data'].to_dataset()
        lat_raw = geo['Latitude'].values
        north_up = float(lat_raw[0].mean()) > float(lat_raw[-1].mean())

        for src, dst in TIE_FIELDS:
            da = geo[src]
            expected = (np.float32(da.attrs['Slope']) * da.values
                        + np.float32(da.attrs['Offset'])).astype('float32')
            if north_up:
                expected = expected[::-1]
            got = l1[dst].values
            assert got.dtype == np.float32, dst
            assert got.shape == expected.shape, dst
            assert np.array_equal(got, expected), dst

        # Tie coordinates are pixel positions (Resampling_interval = 10).
        for dim in (str(names.rows) + '_tie', str(names.columns) + '_tie'):
            assert np.array_equal(l1[dim].values,
                                  np.arange(l1.sizes[dim]) * 10), dim
    finally:
        tree.close()


def test_l1b_rtoa_formula(l1, l1_dir):
    """Rtoa = ((Lt & Mask) * Slope_refl + Offset_refl) / cos(sza), per band.

    The v1 reader's Rtoa divided by the cosine of its inflated sza and cannot
    be pinned; the product's raw counts are the reference instead.
    """
    rows, cols = str(names.rows), str(names.columns)
    tree = xr.open_datatree(l1_dir, engine='h5netcdf', phony_dims='sort')
    try:
        im = tree['Image_data'].to_dataset()
        geo = tree['Geometry_data'].to_dataset()
        lat_raw = geo['Latitude'].values
        north_up = float(lat_raw[0].mean()) > float(lat_raw[-1].mean())

        ny = l1.sizes[rows]
        r_file = [ny - 1 - r for r in WIN_ROWS] if north_up else list(WIN_ROWS)

        sza = l1[str(names.sza)].isel({rows: WIN_ROWS, cols: WIN_COLS}) \
                               .compute().values
        mus = np.cos(np.radians(sza))

        got = l1[str(names.rtoa)].isel({rows: WIN_ROWS, cols: WIN_COLS}) \
                               .compute().values              # (bands, 3, 3)
        assert got.dtype == np.float32
        for band_i in (0, 4, 10):
            band = l1.coords[str(names.bands)].values[band_i]
            lt = im[f'Lt_{band[:4]}']
            counts = lt.values[np.ix_(r_file, WIN_COLS)]
            expected = (((counts & np.uint16(lt.attrs['Mask']))
                         * lt.attrs['Slope_reflectance']
                         + lt.attrs['Offset_reflectance']) / mus)
            assert np.allclose(got[band_i], expected, rtol=1e-7), band
    finally:
        tree.close()


def test_l1b_lazy_load(l1):
    generic.Test.lazy_load(l1)


def test_l1b_flagreader(l1):
    generic.Test.flagreader(l1)


def test_l1b_flag_values(l1):
    """Flag bits decode to the product's own masks (100x100 slice)."""
    sub = l1.isel({str(names.rows): slice(500, 600), str(names.columns): slice(300, 400)})
    flags = FlagsInit({GenericFlags.L1_INVALID: 4, GenericFlags.LAND: 8},
                      'uint8', l1.attrs['_flag_reader'])
    f = flags.map_blocks(sub)['flags'].compute().values

    qf = sub['quality_flag'].compute().values
    water = sub['water'].compute().values
    # Bit 0 set = "Integrity" (sound) per the product's own description; the
    # v1 raw-value test flagged the sound ~86% of this scene.
    assert np.array_equal((f & 4) != 0, (qf & 1) == 0)
    # water is a land percentage 0-100 (255 = error); LAND at the majority
    # threshold. v1's `~water` on the uint8 mask was set on every pixel.
    assert np.array_equal((f & 8) != 0, water >= 50)

    # Scene facts (pinned sample): both quality states occur, and the land
    # percentage spans pure water, mixed and pure land pixels — so neither
    # comparison above is vacuous.
    assert set(np.unique(l1['quality_flag'].values)) == {0, 1}
    w = l1['water'].values
    assert (w == 0).any() and ((w > 0) & (w < 100)).any() and (w == 100).any()


def test_l1b_subset(l1_dir):
    ds = Level1B_SGLI(l1_dir, metadata_template=[])
    generic.Test.subset(ds)


def test_l1b_plot(request, l1):
    band = str(l1.coords[str(names.bands)].values[4])   # VN05 (530 nm)
    generic.plot(request, l1, band)


# --------------------------------------------------------------------------
# Band table and attributes
# --------------------------------------------------------------------------

def test_l1b_bands_cwav(l1, l1_dir):
    """Band labels from the product's Stored_channels; cwav is the hardcoded
    registry table, kept from v1."""
    tree = xr.open_datatree(l1_dir, engine='h5netcdf', phony_dims='sort')
    try:
        labels = dict(tree['Global_attributes'].attrs)['Stored_channels'].split(',')
    finally:
        tree.close()

    got = list(l1.coords[str(names.bands)].values)
    assert got == labels
    assert len(got) == 11 and got[0] == 'VN01(S06)' and got[-1] == 'VN11(S10)'

    cwav = l1[str(names.cwav)].values
    assert cwav.dtype == np.float32
    assert np.array_equal(cwav, np.array(SGLI_CENTRAL_WAVELENGTHS, dtype='float32'))

    bg = list(l1.coords[str(names.bgroup)].values)
    assert bg == ['bands'] * 11


def test_l1b_attrs(l1, l1_dir):
    a = l1.attrs
    assert a[str(names.sensor)] == 'SGLI'
    assert a[str(names.platform)] == 'GCOM-C'
    assert a[str(names.product_name)] == Path(l1_dir).name
    assert a[str(names.input_directory)] == str(Path(l1_dir).parent)
    # No data-driven spacing field exists in the product; the v1 hardcoded
    # value is kept.
    assert a[str(names.resolution)] == 250
    assert a[str(names.shortname)] == '1BSG'
    assert a[str(names.crs)] == 'EPSG:4326'

    md = a['metadata']
    # The file stores a non-ISO scene time; v2 normalizes it (same instant).
    assert a[str(names.datetime)] == parse(md['Scene_center_time']).isoformat()
    assert md['Product_level'] == 'Level-1B'
    assert a['_flag_reader'] == 'eoread.readers.sgli.Level1B_SGLI'
    # SRF getter attributes carried over from v1 (consumed by eotools).
    assert a['_srf_getter'] == 'eotools.srf.get_SRF_eumetsat'
    assert a['_srf_getter_arg'] == 'gcom-c_1_sgli'


# --------------------------------------------------------------------------
# v1 compatibility
# --------------------------------------------------------------------------

def test_l1b_v1_compat(l1):
    """SGLI is an identity transform: v1 strips schema_version, keeps the rest.

    Values differ from the v1 output by the documented fixes (south-up flip,
    affine scale factors — see REFACTORING.md); this pins layout.
    """
    ds = v1(l1)
    assert 'schema_version' not in ds.attrs
    for var in (str(names.rtoa), 'quality_flag', 'water',
                'lat_tie', 'lon_tie', str(names.vza) + '_tie'):
        assert var in ds
    assert list(ds.coords[str(names.bands)].values) == \
        list(l1.coords[str(names.bands)].values)
    # Superset attributes pass through untouched.
    assert ds.attrs[str(names.shortname)] == '1BSG'
    assert ds.attrs[str(names.crs)] == 'EPSG:4326'


# --------------------------------------------------------------------------
# Detection and error paths
# --------------------------------------------------------------------------

def test_variant_detection():
    reader = object.__new__(Level1B_SGLI)
    assert reader.detect_variant({'Product_level': 'Level-1B'}) == 'l1b'
    with pytest.raises(UnsupportedProductType):
        reader.detect_variant({'Product_level': 'Level-1A'})


def test_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        Level1B_SGLI(tmp_path / 'nope.h5')
