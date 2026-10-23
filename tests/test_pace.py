#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance + legacy parity tests for the PACE OCI reader (L1B).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

Parity with the legacy reader (eoread.legacy.pace) is pinned at a fixed
window of pixel coordinates. Documented deviations from legacy (see
REFACTORING.md): no orientation change (the product is natively south-up),
per-band chunking, boolean flag masks where legacy returned raw values (and
would raise on the float32 watermask for LAND), plus the added shortname/crs
attributes.

The sample is a curved-swatch scene: latitude varies by several degrees along
a raster row, so `generic.Test.latlon`'s flat-projection check does not
apply; and the azimuth angles are signed relative values in [-180, 180] while
the naming registry declares [0, 180], so `generic.Test.angles` does not
either. Both are replaced by scene-aware checks below.
"""

import gc
import re
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
from eoread.readers.pace import Level1B_PACE_OCI
from eoread.legacy.pace import Level1B_PACE_OCI as Level1B_PACE_OCI_legacy

# Fixed comparison window: raster rows x columns, well inside the 1709x1272 scene.
WIN_ROWS = [400, 850, 1300]
WIN_COLS = [200, 640, 1080]

# Every full-grid variable (canonical names; legacy uses the same).
TWO_D_VARS = ('latitude', 'longitude', 'height', 'water', 'quality_flag',
              'sza', 'saa', 'vza', 'vaa')


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so transient legacy builds do not accumulate across the session.
    yield
    gc.collect()


@pytest.fixture(scope='session')
def l1_dir():
    return get_sample(Level1B_PACE_OCI)


@pytest.fixture(scope='session')
def l1(l1_dir):
    return Level1B_PACE_OCI(l1_dir, chunks=500)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------

def test_l1b_instantiation(l1_dir, chunks):
    Level1B_PACE_OCI(l1_dir, chunks=chunks)


def test_l1b_main(l1):
    generic.Test.main(l1, angle_data=True)


def test_l1b_conformance(l1):
    generic.Test.v2_conformance(l1, primary='Rtoa')


def test_l1b_latlon(l1):
    """Geolocation checks against the product's own footprint.

    Deliberately not `generic.Test.latlon`: this is a curved swath, latitude
    varies by ~2.5 deg along a raster row (std 1.25 on row 0), so the
    flat-projection invariance check would raise on valid data.
    """
    rows, cols = str(names.rows), str(names.columns)

    # Along-track orientation: col-0 latitude runs south to north; at most one
    # turn is allowed (polar orbits dip back around the pole).
    prof = l1[str(names.lat)].isel({cols: 0}).values
    s = np.sign(np.diff(prof))
    nz = s[s != 0]
    n_max = int(((nz[:-1] > 0) & (nz[1:] < 0)).sum())
    n_min = int(((nz[:-1] < 0) & (nz[1:] > 0)).sum())
    assert prof[0] < prof[-1], 'col-0 latitude must run south to north'
    assert n_max == 0 and n_min <= 1, \
        f'unexpected turns in col-0 latitude ({n_max} maxima, {n_min} minima)'

    # Full-grid extent vs the ACDD geospatial_bounds polygon corners (rounded
    # to 5 decimals in the file). The scalar geospatial_lon_max attribute is
    # not used: on this product it underestimates the true raster maximum by
    # ~0.14 deg.
    bounds = l1.attrs['metadata']['geospatial_bounds']
    lats, lons = zip(*re.findall(r'(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)', bounds))
    lats = [float(v) for v in lats]
    lons = [float(v) for v in lons]
    assert abs(l1[str(names.lat)].values.min() - min(lats)) < 1e-3
    assert abs(l1[str(names.lat)].values.max() - max(lats)) < 1e-3
    assert abs(l1[str(names.lon)].values.min() - min(lons)) < 1e-3
    assert abs(l1[str(names.lon)].values.max() - max(lons)) < 1e-3


def test_l1b_angles(l1):
    """Angle ranges; deliberately not `generic.Test.angles`.

    The product stores signed relative azimuths in [-180, 180] (reaching
    ~±179.99 on this scene) while the naming registry declares [0, 180] — a
    documented registry inconsistency (OLCI/MSI/MERIS precedent). The zenith
    angles are absolute and stay within [0, 90].
    """
    for var, lo, hi in ((str(names.sza), 0., 90.), (str(names.vza), 0., 90.),
                        (str(names.saa), -180., 180.), (str(names.vaa), -180., 180.)):
        v = l1[var].isel({str(names.rows): slice(None, None, 50),
                          str(names.columns): slice(None, None, 50)}).values
        assert v.min() >= lo - 1e-3 and v.max() <= hi + 1e-3, \
            f'{var} out of range [{lo}, {hi}], got [{v.min()}, {v.max()}]'


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
    assert np.array_equal((f & 4) != 0, qf != 0)
    assert np.array_equal((f & 8) != 0, ~water.astype(bool))

    # Scene facts: this product carries no invalid pixels; land and water both
    # occur (so the LAND comparison above is non-vacuous).
    assert float(l1['quality_flag'].values.max()) == 0.0
    w = l1['water'].values
    assert (w == 0).any() and (w == 1).any()


def test_l1b_subset(l1_dir):
    ds = Level1B_PACE_OCI(l1_dir, metadata_template=[])
    generic.Test.subset(ds)


def test_l1b_plot(request, l1):
    generic.plot(request, l1, '892')


# --------------------------------------------------------------------------
# Band table and attributes
# --------------------------------------------------------------------------

def test_l1b_bands_cwav(l1, l1_dir):
    """Band table: dedup + lexicographic label sort, recomputed from the product."""
    # Same engine as the reader: keep the process on h5py's libhdf5 only.
    tree = xr.open_datatree(l1_dir, engine='h5netcdf')
    try:
        bdata = tree['sensor_band_parameters'].to_dataset()
        keys = ('blue_wavelength', 'red_wavelength', 'SWIR_wavelength')
        wls = np.concatenate([bdata[k].values for k in keys])
        groups = (['blue_bands'] * len(bdata['blue_wavelength']) +
                  ['red_bands'] * len(bdata['red_wavelength']) +
                  ['SWIR_bands'] * len(bdata['SWIR_wavelength']))
    finally:
        tree.close()

    labels = wls.astype(int).astype(str)
    seen = set()
    kept = [i for i, w in enumerate(labels) if w not in seen and not seen.add(w)]
    order = sorted(kept, key=lambda i: labels[i])
    seq = [labels[i] for i in order]

    got = list(l1.coords[str(names.bands)].values)
    assert len(got) == 289
    assert got == seq
    # The legacy sort key is the string label, not the wavelength: four-digit
    # SWIR labels come first and the sequence is lexicographically sorted.
    assert got == sorted(got)
    assert got[0] == '1038' and got[-1] == '939'

    cwav = l1[str(names.cwav)].values
    assert cwav.dtype == np.float32
    assert np.array_equal(cwav, wls[order])

    bg = list(l1.coords[str(names.bgroup)].values)
    assert bg == [groups[i] for i in order]


def test_l1b_attrs(l1, l1_dir):
    a = l1.attrs
    assert a[str(names.sensor)] == 'OCI'
    assert a[str(names.platform)] == 'PACE'
    assert a[str(names.product_name)] == Path(l1_dir).name
    assert a[str(names.input_directory)] == str(Path(l1_dir).parent)
    # No data-driven spacing field exists in the product; legacy's hardcoded
    # value is kept.
    assert a[str(names.resolution)] == 1200
    assert a[str(names.shortname)] == 'PACE_OCI'
    assert a[str(names.crs)] == 'EPSG:4326'

    # datetime is the midpoint of the product's time coverage, exactly as
    # legacy computed it.
    md = a['metadata']
    start, end = parse(md['time_coverage_start']), parse(md['time_coverage_end'])
    assert a[str(names.datetime)] == (start + (end - start) / 2).isoformat()

    # metadata carries the raw root attributes.
    assert md['instrument'] == 'OCI' and md['processing_level'] == 'L1B'
    assert a['_flag_reader'] == 'eoread.readers.pace.Level1B_PACE_OCI'


# --------------------------------------------------------------------------
# v1 compatibility and legacy parity
# --------------------------------------------------------------------------

def test_l1b_v1_compat(l1):
    """OCI is an identity transform: v1 strips schema_version, keeps the rest."""
    ds = v1(l1)
    assert 'schema_version' not in ds.attrs
    for var in (str(names.rtoa), 'quality_flag', 'water'):
        assert var in ds
    assert list(ds.coords[str(names.bands)].values) == \
        list(l1.coords[str(names.bands)].values)
    # Superset attributes pass through untouched.
    assert ds.attrs[str(names.shortname)] == 'PACE_OCI'
    assert ds.attrs[str(names.crs)] == 'EPSG:4326'


def test_l1b_parity_legacy(l1, l1_dir, monkeypatch):
    """Bit-exact parity with the legacy reader at a pinned window."""
    # The v2 reader already opened the file with h5netcdf in this process; the
    # legacy reader's default netCDF4 open then fails with a NetCDF HDF error
    # (two bundled libhdf5 copies resident, see REFACTORING.md). Force the
    # legacy open onto h5netcdf too so the comparison can run in one process.
    import eoread.legacy.pace as legacy_pace
    real_open = legacy_pace.xr.open_datatree

    def _open_h5netcdf(*args, **kwargs):
        kwargs.setdefault('engine', 'h5netcdf')
        return real_open(*args, **kwargs)

    monkeypatch.setattr(legacy_pace.xr, 'open_datatree', _open_h5netcdf)
    legacy = Level1B_PACE_OCI_legacy(l1_dir, chunks=500, verbose=False)
    try:
        r, c = WIN_ROWS, WIN_COLS

        # Payload: all bands at the window.
        assert np.array_equal(
            l1[str(names.rtoa)].isel(y=r, x=c).compute().values,
            legacy['Rtoa'].isel(y=r, x=c).compute().values)

        # Every full-grid variable at the window.
        for var in TWO_D_VARS:
            assert np.array_equal(l1[var].isel(y=r, x=c).compute().values,
                                  legacy[var].isel(y=r, x=c).compute().values), var

        # Band table in full.
        assert np.array_equal(l1[str(names.cwav)].values, legacy['cwav'].values)
        assert list(l1.coords[str(names.bands)].values) == \
            list(legacy.coords['bands'].values)
        assert list(l1.coords[str(names.bgroup)].values) == \
            list(legacy.coords['bands_group'].values)

        # Shared attributes.
        for key in (str(names.sensor), str(names.platform), str(names.product_name),
                    str(names.input_directory), str(names.datetime)):
            assert l1.attrs[key] == legacy.attrs[key], key
        assert l1.attrs[str(names.resolution)] == 1200
        assert legacy.attrs[str(names.resolution)] == 1200

        # Superset attributes: absent in legacy.
        for key in (str(names.shortname), str(names.crs)):
            assert key not in legacy.attrs
    finally:
        del legacy
        gc.collect()


# --------------------------------------------------------------------------
# Detection and error paths
# --------------------------------------------------------------------------

def test_variant_detection():
    reader = object.__new__(Level1B_PACE_OCI)
    assert reader.detect_variant({'processing_level': 'L1B'}) == 'l1b'
    with pytest.raises(UnsupportedProductType):
        reader.detect_variant({'processing_level': 'L1A'})


def test_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        Level1B_PACE_OCI(tmp_path / 'nope.nc')
