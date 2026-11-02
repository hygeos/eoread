#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the GOES-NG ABI L1b reader (ICARE/CMS ``Emultic``).

Real-data policy (tests/README.md): the sample comes from the registry via
`get_sample`; a missing sample means a clean skip. The sample is a directory of
symlinks holding the 1 km and 500 m Emultic files, the ICARE static grid and
the NWC/GEO cloud mask.

The v1 GOES-NG reader was unusable (it referenced undefined globals) and is
removed, so the values are pinned against the raw NetCDF files directly.
"""

import gc
from pathlib import Path

import numpy as np
import pysolar.solar as pysol
import pytest
import xarray as xr
from core.geo.naming import names
from dateutil.parser import parse

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.flags import FlagsInit, GenericFlags
from eoread.reader import ProductError
from eoread.readers.goesng import BANDS, CWAV, Level1B_GOESNG

#: Pinned comparison window in canonical (south-up) rows/columns; well inside
#: the full disk and on valid, cloud-free-ish pixels.
WIN = (5000, 5000)

#: Data values at WIN, pinned from the raw NetCDF files (reflectance).
PINNED_RTOA = (0.1291, 0.08385, 0.1485, 0.1142)
PINNED_LATLON = (-3.8439548, -78.8268356)
PINNED_VZA, PINNED_VAA = 6.3861156, 44.9362526
PINNED_SZA, PINNED_SAA = 53.932356, 57.239119
PINNED_AREA = 1.0071889


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so reader datasets are released promptly.
    yield
    gc.collect()


@pytest.fixture(scope='session')
def sample():
    return get_sample(Level1B_GOESNG)


@pytest.fixture(params=[1000, (400, 600)])
def chunks(request):
    return request.param


@pytest.fixture
def ds(sample, chunks):
    return Level1B_GOESNG(sample, chunks=chunks)


@pytest.fixture
def ds_cloud(sample):
    return Level1B_GOESNG(sample, cloudmask=True)


def _raw_1km(sample):
    return xr.open_dataset(sample, engine='h5netcdf')


def _raw_500m(sample):
    return xr.open_dataset(
        str(sample).replace('Emultic1kmNC4', 'Emultic500mNC4'),
        engine='h5netcdf')


def _raw_aux(sample):
    return xr.open_dataset(sample.with_name('GOESNG-0750.1km.nc'),
                           engine='h5netcdf')


def _raw_cma(sample):
    return xr.open_dataset(
        sample.with_name('S_NWC_CMA_GOES16_globeE-NR_20220617T140000Z.nc'),
        engine='h5netcdf')


def _raw_row(ds, row):
    """Canonical row -> stored (north-up) row."""
    return ds.sizes[str(names.rows)] - 1 - row


def _staged_dir(tmp_path, sample, *, bands_500m=True, aux=True, cma=False):
    """Recreate the sample staging in tmp_path with selected companions."""
    staged = tmp_path / 'GOESNG'
    staged.mkdir()
    (staged / sample.name).symlink_to(sample)
    if bands_500m:
        (staged / sample.name.replace('Emultic1kmNC4', 'Emultic500mNC4')).symlink_to(
            sample.with_name(sample.name.replace('Emultic1kmNC4', 'Emultic500mNC4')))
    if aux:
        (staged / 'GOESNG-0750.1km.nc').symlink_to(
            sample.with_name('GOESNG-0750.1km.nc'))
    if cma:
        (staged / 'S_NWC_CMA_GOES16_globeE-NR_20220617T140000Z.nc').symlink_to(
            sample.with_name('S_NWC_CMA_GOES16_globeE-NR_20220617T140000Z.nc'))
    return staged / sample.name


# ===========================================================================
# Contract
# ===========================================================================

def test_instantiation(sample):
    Level1B_GOESNG(sample)


def test_main(ds):
    generic.Test.main(ds, angle_data=True)


def test_conformance(ds):
    generic.Test.v2_conformance(ds, primary=str(names.rtoa))


def test_latlon(ds):
    # Geostationary projection: lat/lon have no flat-projection invariance.
    generic.Test.latlon(ds, flat_proj=False)


def test_orientation(ds):
    """Canonical south-up / west-to-east orientation."""
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    # The disk edge is NaN (space): compare the first/last valid entries.
    lat = ds[str(names.lat)].isel({str(names.columns): nx // 2}).compute().values
    lat = lat[~np.isnan(lat)]
    assert lat[0] < lat[-1], 'latitude must increase with the row index'
    row = ds[str(names.lon)].isel({str(names.rows): ny // 2}).compute().values
    row = row[~np.isnan(row)]
    assert row[0] < row[-1], 'longitude must increase with the column index'
    assert list(ds[str(names.rows)].values) == list(range(ny))
    assert list(ds[str(names.columns)].values) == list(range(nx))


def test_bands(ds):
    assert list(ds[str(names.bands)].values) == list(BANDS)
    assert np.array_equal(ds[str(names.cwav)].values, np.asarray(CWAV))
    assert ds[str(names.cwav)].attrs['unit'] == 'nm'
    assert set(ds[str(names.bgroup)].values) == {'bands_vnir'}
    assert ds[str(names.rtoa)].dtype == np.dtype('float32')
    assert ds[str(names.rtoa)].attrs['unit'] == '1'


def test_lazy_load(ds):
    generic.Test.lazy_load(ds)


def test_subset(ds):
    generic.Test.subset(ds)


def test_plot(request, ds):
    # The full disk is 10848 x 10848 and the generic plot materializes every
    # field: plot a central 1000 x 1000 window instead (same rendering path,
    # no multi-hundred-MB full-grid compute).
    rows, cols = str(names.rows), str(names.columns)
    y0 = x0 = 4500
    size = 1000
    sub = ds.isel({rows: slice(y0, y0 + size),
                   cols: slice(x0, x0 + size)})
    # `poi` is positional on the subset; the plotted coordinates stay the
    # full-disk labels.
    generic.plot(request, sub, 'VIS_006',
                 poi={rows: size // 2, cols: size // 2})


def test_v1_compat(ds):
    out = v1(ds)
    assert 'schema_version' not in out.attrs
    assert str(names.rtoa) in out
    assert list(out.coords[str(names.bands)].values) == list(BANDS)
    assert out.attrs[str(names.platform)] == 'GOES-16'


# ===========================================================================
# Values pinned against the raw files
# ===========================================================================

def test_payload_values(ds):
    rtoa = ds[str(names.rtoa)].isel(
        {str(names.rows): WIN[0], str(names.columns): WIN[1]}).compute().values
    np.testing.assert_allclose(rtoa, PINNED_RTOA, rtol=0, atol=1e-6)


def test_payload_parity_raw(ds, sample):
    """Each band equals the raw file (1 km) or its 2x2 average (500 m)."""
    ny = ds.sizes[str(names.rows)]
    raw = _raw_1km(sample)
    win = np.array([100, 4000, ny - 101])
    cols = np.array([100, 4000, 9000])

    for band in ('VIS_004', 'VIS_008', 'VIS_016'):
        got = ds[str(names.rtoa)].sel({str(names.bands): band}).isel(
            {str(names.rows): win, str(names.columns): cols}).compute().values
        expected = raw[band].isel(
            ny1km=ny - 1 - win, nx1km=cols).values / 100.
        np.testing.assert_allclose(got, expected.astype('float32'),
                                   rtol=0, atol=1e-7, equal_nan=True)

    raw5 = _raw_500m(sample)
    expected = np.mean([
        raw5['VIS_006'].isel(ny500m=2 * (ny - 1 - win) + i,
                             nx500m=2 * cols + j).values
        for i in (0, 1) for j in (0, 1)], axis=0) / 100.
    got = ds[str(names.rtoa)].sel({str(names.bands): 'VIS_006'}).isel(
        {str(names.rows): win, str(names.columns): cols}).compute().values
    np.testing.assert_allclose(got, expected.astype('float32'),
                               rtol=0, atol=1e-7, equal_nan=True)


def test_geometry_parity_aux(ds, sample):
    ny = ds.sizes[str(names.rows)]
    win = np.array([100, 5000, ny - 101])
    cols = np.array([100, 5000, 10000])
    aux = _raw_aux(sample)
    for source, dest in (('Latitude', names.lat), ('Longitude', names.lon),
                         ('View_Zenith', names.vza), ('View_Azimuth', names.vaa)):
        got = ds[str(dest)].isel({str(names.rows): win,
                                  str(names.columns): cols}).compute().values
        expected = aux[source].isel(Nlin=ny - 1 - win, Ncol=cols).values
        np.testing.assert_array_equal(got, expected.astype('float32'))


def test_geometry_pins(ds):
    y, x = WIN
    got = {str(v): float(ds[str(v)].isel(
        {str(names.rows): y, str(names.columns): x}).compute())
        for v in (names.lat, names.lon, names.vza, names.vaa)}
    got['pixel_area_size'] = float(ds['pixel_area_size'].isel(
        {str(names.rows): y, str(names.columns): x}).compute())
    assert got[str(names.lat)] == pytest.approx(PINNED_LATLON[0], abs=1e-6)
    assert got[str(names.lon)] == pytest.approx(PINNED_LATLON[1], abs=1e-6)
    assert got[str(names.vza)] == pytest.approx(PINNED_VZA, abs=1e-5)
    assert got[str(names.vaa)] == pytest.approx(PINNED_VAA, abs=1e-5)
    assert got['pixel_area_size'] == pytest.approx(PINNED_AREA, abs=1e-6)


def test_angles(ds):
    """Solar and viewing angles are physical and wired to the right pixels."""
    y, x = WIN
    for name, expected in ((names.sza, PINNED_SZA), (names.saa, PINNED_SAA)):
        got = float(ds[str(name)].isel({str(names.rows): y,
                                        str(names.columns): x}).compute())
        assert got == pytest.approx(expected, abs=1e-4)

    # Range check on a valid sub-window (the disk edge is NaN).
    sub = {str(a): ds[str(a)].isel({str(names.rows): slice(5000, 5050),
                                    str(names.columns): slice(5000, 5050)}).compute().values
           for a in (names.sza, names.saa, names.vza, names.vaa)}
    assert np.nanmin(sub[str(names.sza)]) >= 0. and np.nanmax(sub[str(names.sza)]) <= 90.
    assert np.nanmin(sub[str(names.vza)]) >= 0. and np.nanmax(sub[str(names.vza)]) <= 90.
    for name in (names.saa, names.vaa):
        assert np.nanmin(sub[str(name)]) >= 0. and np.nanmax(sub[str(name)]) <= 360.


def test_solar_angles_parity_pysolar(ds, sample):
    """sza/saa reproduce pysolar at the file's acquisition midpoint."""
    raw = _raw_1km(sample)
    start = parse(raw.attrs['time_coverage_start'])
    stop = parse(raw.attrs['time_coverage_end'])
    dt = start + (stop - start) / 2
    y, x = WIN
    lat = ds[str(names.lat)].isel({str(names.rows): y,
                                   str(names.columns): x}).compute().values.item()
    lon = ds[str(names.lon)].isel({str(names.rows): y,
                                   str(names.columns): x}).compute().values.item()
    expected_sza = 90. - pysol.get_altitude(lat, lon, dt, pressure=0., elevation=0.)
    expected_saa = pysol.get_azimuth(lat, lon, dt)
    assert float(ds[str(names.sza)].isel(
        {str(names.rows): y, str(names.columns): x}).compute()) \
        == pytest.approx(expected_sza, abs=1e-4)
    assert float(ds[str(names.saa)].isel(
        {str(names.rows): y, str(names.columns): x}).compute()) \
        == pytest.approx(expected_saa, abs=1e-4)


# ===========================================================================
# Cloud mask and flags
# ===========================================================================

def test_cloudmask(ds_cloud, sample):
    assert 'cloud' in ds_cloud
    assert ds_cloud['cloud'].dims == (str(names.rows), str(names.columns))
    assert ds_cloud['cloud'].dtype == np.dtype('float32')
    y, x = WIN
    got = float(ds_cloud['cloud'].isel({str(names.rows): y,
                                       str(names.columns): x}).compute())
    assert np.isnan(got) or got in (0., 1.)

    # 2x2 repetition of the 2 km mask, with the same south-up flip.
    cma = _raw_cma(sample)['cma']
    raw_r = _raw_row(ds_cloud, y)
    expected = cma.isel(ny=raw_r // 2, nx=x // 2).values
    assert got == pytest.approx(expected, nan_ok=True)


def test_flagreader(ds):
    generic.Test.flagreader(ds)


def test_cloud_flag(ds_cloud):
    sub = ds_cloud.isel({str(names.rows): slice(5000, 5050),
                         str(names.columns): slice(5000, 5050)}).compute()
    cloud = Level1B_GOESNG.getflag(sub, GenericFlags.CLOUD)
    assert cloud.dtype == bool
    np.testing.assert_array_equal(cloud.values, sub['cloud'].values == 1)

    flags = FlagsInit({GenericFlags.CLOUD: 1, GenericFlags.L1_INVALID: 4},
                      'uint8', ds_cloud.attrs['_flag_reader']).map_blocks(
        ds_cloud.isel({str(names.rows): slice(5000, 5050),
                       str(names.columns): slice(5000, 5050)}))
    values = flags['flags'].compute().values
    assert set(np.unique(values)) <= {0, 1, 4, 5}


def test_cloud_flag_requires_mask(ds):
    sub = ds.isel({str(names.rows): slice(5000, 5010),
                   str(names.columns): slice(5000, 5010)}).compute()
    with pytest.raises(ValueError, match='cloudmask=True'):
        Level1B_GOESNG.getflag(sub, GenericFlags.CLOUD)


# ===========================================================================
# Failure modes
# ===========================================================================

def test_missing_path(tmp_path):
    with pytest.raises(ProductError, match='does not exist'):
        Level1B_GOESNG(tmp_path / 'missing.nc')


def test_wrong_product(tmp_path, sample):
    wrong = tmp_path / 'not_a_goes_product.nc'
    wrong.symlink_to(sample)
    with pytest.raises(ProductError, match='Emultic1kmNC4'):
        Level1B_GOESNG(wrong)


def test_missing_500m(tmp_path, sample):
    path = _staged_dir(tmp_path, sample, bands_500m=False)
    with pytest.raises(ProductError, match='500 m file'):
        Level1B_GOESNG(path)


def test_missing_aux(tmp_path, sample):
    path = _staged_dir(tmp_path, sample, aux=False)
    with pytest.raises(ProductError, match='no auxiliary grid'):
        Level1B_GOESNG(path)


def test_cloudmask_missing_cma(tmp_path, sample):
    path = _staged_dir(tmp_path, sample, cma=False)
    with pytest.raises(ProductError, match='CMA file'):
        Level1B_GOESNG(path, cloudmask=True)
