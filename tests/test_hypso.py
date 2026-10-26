#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the HYPSO-1 L1B reader.

Real-data policy (tests/README.md): the sample comes from the registry via
`get_sample`; a missing sample means a clean skip. The sample is the NTNU
``vancouver_2022-07-30`` L1B product (956 x 684 pixels, 120 bands,
``navigation`` group).

The v1 reader is removed; its output could not serve as a value reference
anyway (it left the payload uint16 and kept the north-up orientation). Values
are therefore pinned against the raw NetCDF file.
"""

import gc

import numpy as np
import pytest
import xarray as xr
from core.geo.naming import names

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.reader import ProductError
from eoread.readers.hypso import Level1_HYPSO

#: Pinned pixel in canonical (south-up) coordinates.
WIN = (400, 300)

PINNED_LTOA = (0., 42., 115.)          # bands 0, 50, 119
PINNED_LATLON = (48.5876656, -123.9863892)
PINNED_ANGLES = (37.9715385, 99.4598999, 38.2908211, 132.1976318)


@pytest.fixture(autouse=True)
def _gc():
    yield
    gc.collect()


@pytest.fixture(scope='session')
def sample():
    return get_sample(Level1_HYPSO)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


@pytest.fixture
def ds(sample, chunks):
    return Level1_HYPSO(sample, chunks=chunks)


def _raw(sample):
    return xr.open_datatree(sample, engine='h5netcdf')


# ===========================================================================
# Contract
# ===========================================================================

def test_instantiation(sample):
    Level1_HYPSO(sample)


def test_main(ds):
    generic.Test.main(ds, angle_data=True)


def test_conformance(ds):
    generic.Test.v2_conformance(ds, primary=str(names.ltoa))


def test_latlon(ds):
    generic.Test.latlon(ds, flat_proj=False)


def test_orientation(ds):
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    lat = ds[str(names.lat)].isel({str(names.columns): nx // 2}).compute().values
    assert lat[0] < lat[-1], 'latitude must increase with the row index'
    lon = ds[str(names.lon)].isel({str(names.rows): ny // 2}).compute().values
    assert lon[0] < lon[-1], 'longitude must increase with the column index'
    assert list(ds[str(names.rows)].values) == list(range(ny))
    assert list(ds[str(names.columns)].values) == list(range(nx))


def test_bands(ds):
    assert list(ds[str(names.bands)].values) == [str(i) for i in range(120)]
    assert set(ds[str(names.bgroup)].values) == {'bands_vnir'}
    cwav = ds[str(names.cwav)].values
    assert cwav.shape == (120,)
    assert cwav[0] == pytest.approx(387.8) and cwav[-1] == pytest.approx(802.5)
    assert ds[str(names.cwav)].attrs['unit'] == 'nm'
    assert ds[str(names.ltoa)].dtype == np.dtype('float32')
    assert ds[str(names.ltoa)].attrs['unit'] == 'W/m^2/micrometer/sr'


def test_lazy_load(ds):
    generic.Test.lazy_load(ds)


def test_subset(ds):
    generic.Test.subset(ds)


def test_plot(request, ds):
    rows, cols = str(names.rows), str(names.columns)
    sub = ds.isel({rows: slice(300, 700), cols: slice(200, 600)})
    generic.plot(request, sub, '50', poi={rows: 200, cols: 200})


def test_v1_compat(ds):
    out = v1(ds)
    assert 'schema_version' not in out.attrs
    assert str(names.ltoa) in out
    assert out.attrs[str(names.platform)] == 'HYPSO'


def test_no_flag_reader(ds):
    assert '_flag_reader' not in ds.attrs


# ===========================================================================
# Values pinned against the raw file
# ===========================================================================

def test_payload_pins(ds):
    y, x = WIN
    got = ds[str(names.ltoa)].isel({str(names.rows): y,
                                    str(names.columns): x}).compute().values
    np.testing.assert_array_equal(got[[0, 50, 119]], np.asarray(PINNED_LTOA))


def test_payload_parity_raw(ds, sample):
    ny = ds.sizes[str(names.rows)]
    win = np.array([0, 400, ny - 1])
    cols = np.array([0, 300, 683])
    bands = np.array([0, 50, 119])
    raw = _raw(sample)
    expected = raw['products'].to_dataset()['Lt'].isel(
        lines=ny - 1 - win, samples=cols, bands=bands).transpose(
        'bands', 'lines', 'samples').values
    got = ds[str(names.ltoa)].isel(
        {str(names.rows): win, str(names.columns): cols,
         str(names.bands): bands}).compute().values
    np.testing.assert_array_equal(got, expected.astype('float32'))


def test_geometry_parity_raw(ds, sample):
    ny = ds.sizes[str(names.rows)]
    win = np.array([0, 400, ny - 1])
    cols = np.array([0, 300, 683])
    nav = _raw(sample)['navigation'].to_dataset()
    for source, dest in (('latitude', names.lat), ('longitude', names.lon),
                         ('sensor_zenith', names.vza), ('sensor_azimuth', names.vaa),
                         ('solar_zenith', names.sza), ('solar_azimuth', names.saa)):
        got = ds[str(dest)].isel({str(names.rows): win,
                                  str(names.columns): cols}).compute().values
        expected = nav[source].isel(lines=ny - 1 - win, samples=cols).values
        np.testing.assert_array_equal(got, expected.astype('float32'))


def test_geometry_pins(ds):
    y, x = WIN
    assert float(ds[str(names.lat)].isel({str(names.rows): y,
                                          str(names.columns): x}).compute()) \
        == pytest.approx(PINNED_LATLON[0], abs=1e-6)
    assert float(ds[str(names.lon)].isel({str(names.rows): y,
                                          str(names.columns): x}).compute()) \
        == pytest.approx(PINNED_LATLON[1], abs=1e-6)


def test_angles(ds):
    y, x = WIN
    got = [float(ds[str(v)].isel({str(names.rows): y,
                                  str(names.columns): x}).compute())
           for v in (names.vza, names.vaa, names.sza, names.saa)]
    np.testing.assert_allclose(got, PINNED_ANGLES, rtol=0, atol=1e-5)

    # Scene-wide ranges on a decimated grid.
    sub = {str(v): ds[str(v)].isel({str(names.rows): slice(None, None, 20),
                                    str(names.columns): slice(None, None, 20)}).compute().values
           for v in (names.vza, names.vaa, names.sza, names.saa)}
    for zenith in (names.vza, names.sza):
        assert np.nanmin(sub[str(zenith)]) >= 0. and np.nanmax(sub[str(zenith)]) <= 90.
    for azimuth in (names.vaa, names.saa):
        assert np.nanmin(sub[str(azimuth)]) >= 0. and np.nanmax(sub[str(azimuth)]) <= 360.


# ===========================================================================
# Failure modes
# ===========================================================================

def test_missing_path(tmp_path):
    with pytest.raises(ProductError, match='does not exist'):
        Level1_HYPSO(tmp_path / 'missing.nc')
