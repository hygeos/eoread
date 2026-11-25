#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the NASA L1C (OCSSW l2gen) reader and the NASA EarthData download
utilities.

The fixtures are synthetic NetCDF4 files with the OCSSW group layout
(`/sensor_band_parameters`, `/navigation_data`, `/geophysical_data`); no
public offline L1C sample exists, so the tests pin the reader behavior
(orientation, band ordering, masking, flags, attributes) against the raw
arrays instead of a real product. The download utilities are tested with
their external calls (wget / download_url) monkeypatched — no network.
"""

import gc
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hygeos_core.geo.naming import names

from . import generic
from eoread.ancillary import nasa as anc_nasa
from eoread.flags import GenericFlags
from eoread.reader import ProductError
from eoread.readers.nasa import Level1_NASA
from eoread.schema import SCHEMA_VERSION


# ------------------------------------------------------------------ fixtures

NY, NX = 8, 10

#: file-table order: deliberately not wavelength-sorted, and one entry beyond
#: `number_of_reflective_bands` (the slice must ignore it)
WAV_TABLE = np.array([555., 412., 645., 865.])
NREF = 3

LAT0, LON0 = 40., -70.
DLAT, DLON = 0.05, 0.05
TSTART = '2008-04-11T12:45:00.000Z'
TEND = '2008-04-11T12:47:00Z'
EXPECTED_DT = '2008-04-11T12:46:00+00:00'
FILENAME = 'A2008101014500.L1C.nc'

FLAG_MEANINGS = 'LAND NAVFAIL BOWTIEDEL HILT CLOUD'
FLAG_MASKS = '1 2 4 8 16'


def _canonical_arrays():
    """All group arrays in canonical (south-up, west-to-east) order."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    arrays = {
        # lat varies along y only, lon along x only (generic.Test.latlon)
        'latitude': (LAT0 + yy * DLAT).astype('f4'),
        'longitude': (LON0 + xx * DLON).astype('f4'),
    }
    for pos in range(NREF):
        rhot = (pos * 100 + yy * NX + xx).astype('f4') * 0.5 + 10.
        polcor = np.full((NY, NX), float(pos + 2), 'f4')
        rhot[:2, :3] = -999.                       # masked region (rhot <= -100)
        s = int(WAV_TABLE[pos])
        arrays[f'rhot_{s}'] = rhot
        arrays[f'polcor_{s}'] = polcor
    arrays['solz'] = (30 + yy * 0.1 + xx * 0.01).astype('f4')
    arrays['senz'] = (20 + yy * 0.1 + xx * 0.01).astype('f4')
    arrays['sola'] = (45 + yy * 0.1 + xx * 0.01).astype('f4')
    arrays['sena'] = (60 + yy * 0.1 + xx * 0.01).astype('f4')
    flags = np.zeros((NY, NX), 'u2')
    flags[xx % 2 == 0] |= 1                         # LAND
    flags[yy == 0] |= 2                             # NAVFAIL
    flags[xx >= NX - 1] |= 4                        # BOWTIEDEL
    flags[yy >= NY - 2] |= 16                       # CLOUD
    arrays['l2_flags'] = flags
    return arrays


def _flip(arrays, north_first, east_first):
    out = {}
    for k, v in arrays.items():
        if north_first:
            v = np.flipud(v)
        if east_first:
            v = np.fliplr(v)
        out[k] = v
    return out


def _write_l1c(path, *, north_first=False, east_first=False, control_points=False,
               cp_stride=1, instrument='MODIS', platform='Aqua', short_name=None,
               tstart=TSTART, tend=TEND, drop_band=None, table_size=None):
    """Write a synthetic OCSSW-layout L1C NetCDF4 file to `path`."""
    arrays = _flip(_canonical_arrays(), north_first, east_first)
    if drop_band is not None:
        for sfx in (f'rhot_{drop_band}', f'polcor_{drop_band}'):
            arrays.pop(sfx)

    attrs = {
        'instrument': instrument,
        'platform': platform,
        'time_coverage_start': tstart,
        'time_coverage_end': tend,
        'source': 'OCSSW',
    }
    if short_name is not None:
        attrs['short_name'] = short_name
    xr.Dataset(attrs=attrs).to_netcdf(str(path), mode='w', engine='netcdf4')

    table = WAV_TABLE[: (table_size if table_size is not None else WAV_TABLE.size)]
    sbp = xr.Dataset({
        'wavelength': (('wavelength',), table.astype('f8')),
        'number_of_reflective_bands': ((), np.int32(NREF)),
    })
    sbp.to_netcdf(str(path), mode='a', group='sensor_band_parameters',
                  engine='netcdf4')

    cols = 'pixel_control_points' if control_points else 'pixels_per_line'
    navi = xr.Dataset({
        'latitude': (('number_of_lines', cols), arrays['latitude'][:, ::cp_stride]),
        'longitude': (('number_of_lines', cols), arrays['longitude'][:, ::cp_stride]),
    })
    if control_points:
        navi['cntl_pt_cols'] = (cols, np.arange(1, NX + 1, cp_stride))
        navi['cntl_pt_rows'] = ('number_of_lines', np.arange(1, NY + 1))
    navi.to_netcdf(str(path), mode='a', group='navigation_data', engine='netcdf4')

    geo = xr.Dataset({
        k: (('number_of_lines', 'pixels_per_line'), v)
        for k, v in arrays.items() if k not in ('latitude', 'longitude')})
    geo['l2_flags'].attrs.update({'flag_meanings': FLAG_MEANINGS,
                                  'flag_masks': FLAG_MASKS})
    geo.to_netcdf(str(path), mode='a', group='geophysical_data', engine='netcdf4')


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles, so reader datasets are only freed
    # by the cyclic GC. Force a collection after every test (see test_msi.py).
    yield
    gc.collect()


@pytest.fixture(params=[(False, False), (True, False), (False, True), (True, True)])
def l1c(request, tmp_path):
    north_first, east_first = request.param
    path = tmp_path / FILENAME
    _write_l1c(path, north_first=north_first, east_first=east_first)
    return Level1_NASA(path)


# -------------------------------------------------------------------- reader

def test_main(l1c):
    generic.Test.main(l1c, angle_data=True)
    generic.Test.latlon(l1c)
    generic.Test.angles(l1c)
    generic.Test.lazy_load(l1c)
    generic.Test.v2_conformance(l1c)


def test_rtoa_values(l1c):
    """Rtoa = rhot/polcor masked where rhot <= -100, in canonical orientation.

    Output band i is the file table entry with the i-th smallest wavelength:
    WAV_TABLE[:3] = [555, 412, 645] -> B1 = pos 1 (412), B2 = pos 0 (555),
    B3 = pos 2 (645). The masked corner must land back at canonical [0:2, :3]
    whatever the stored orientation.
    """
    yy, xx = np.mgrid[0:NY, 0:NX]
    for i, pos in enumerate([1, 0, 2]):
        rhot = (pos * 100 + yy * NX + xx).astype('f4') * 0.5 + 10.
        expected = rhot / np.full((NY, NX), float(pos + 2), 'f4')
        expected[(yy < 2) & (xx < 3)] = np.nan
        got = l1c[str(names.rtoa)].isel({str(names.bands): i}).values
        np.testing.assert_allclose(got, expected, rtol=1e-6, equal_nan=True)


def test_latlon_angles(l1c):
    yy, xx = np.mgrid[0:NY, 0:NX]
    np.testing.assert_array_equal(
        l1c[str(names.lat)].values, (LAT0 + yy * DLAT).astype('f4'))
    np.testing.assert_array_equal(
        l1c[str(names.lon)].values, (LON0 + xx * DLON).astype('f4'))
    for var, base in (('sza', 30), ('vza', 20), ('saa', 45), ('vaa', 60)):
        np.testing.assert_array_equal(
            l1c[var].values, (base + yy * 0.1 + xx * 0.01).astype('f4'))


def test_bands(l1c):
    assert [str(v) for v in l1c[str(names.bands)].values] == ['B1', 'B2', 'B3']
    np.testing.assert_allclose(l1c[str(names.cwav)].values, [412., 555., 645.])
    assert l1c[str(names.cwav)].attrs['unit'] == 'nm'
    assert [str(v) for v in l1c[str(names.bgroup)].values] == ['bands_vnir'] * 3


def test_attrs(l1c):
    a = l1c.attrs
    assert a[str(names.sensor)] == 'MODIS'
    assert a[str(names.platform)] == 'Aqua'
    assert a[str(names.product_name)] == FILENAME[:-3]
    assert a[str(names.datetime)] == EXPECTED_DT
    # input_directory is the file parent (the fixture tmp dir name varies):
    # product_name + suffix must reconstruct the original filename
    assert a[str(names.product_name)] == Path(FILENAME).stem
    indir = Path(a[str(names.input_directory)])
    assert (indir / (a[str(names.product_name)] + Path(FILENAME).suffix)).name \
        == FILENAME
    # data-driven resolution: mirror of Level1_NASA._resolution on this grid
    la = (LAT0 + np.arange(NY) * DLAT).astype('f4')[NY // 2]
    lo = (LON0 + np.arange(NX) * DLON).astype('f4')
    dlat = np.diff(np.full(NX, la)) * 111320.
    midlat = float(np.nanmean(np.full(NX, la)))
    dlon = np.diff(lo) * 111320. * np.cos(np.deg2rad(midlat))
    expected_res = int(round(float(np.median(np.hypot(dlat, dlon)))))
    assert a[str(names.resolution)] == expected_res
    assert not isinstance(a[str(names.resolution)], bool)
    assert a[str(names.shortname)] == 'L1C_MODIS'          # no short_name in file
    assert a[str(names.crs)] == 'EPSG:4326'
    assert a['_srf_getter'] == 'eotools.srf.get_SRF_NASA'
    assert a['_srf_getter_arg'] == 'aqua_modis'
    assert a['_flag_reader'] == 'eoread.readers.nasa.Level1_NASA'
    assert a['schema_version'] == SCHEMA_VERSION
    # raw root attrs are curated into `metadata`, not left at top level
    assert a['metadata']['source'] == 'OCSSW'
    assert a['metadata']['time_coverage_start'] == TSTART
    assert 'instrument' not in a


def test_flags(l1c):
    yy, xx = np.mgrid[0:NY, 0:NX]
    land = Level1_NASA.getflag(l1c, GenericFlags.LAND).values
    assert (land == (xx % 2 == 0)).all()
    invalid = Level1_NASA.getflag(l1c, GenericFlags.L1_INVALID).values
    assert (invalid == ((yy == 0) | (xx >= NX - 1))).all()
    with pytest.raises(ValueError):
        Level1_NASA.getflag(l1c, GenericFlags.CLOUD)


def test_flagreader(l1c):
    generic.Test.flagreader(l1c)


def test_control_points_equivalent(tmp_path):
    """One-control-point-per-pixel navigation reads identically to per-pixel."""
    pp = tmp_path / 'per_pixel.nc'
    cp = tmp_path / 'control_points.nc'
    _write_l1c(pp)
    _write_l1c(cp, control_points=True)
    ds1, ds2 = Level1_NASA(pp), Level1_NASA(cp)
    for var in (str(names.lat), str(names.lon), str(names.rtoa)):
        np.testing.assert_allclose(ds1[var].values, ds2[var].values,
                                   rtol=0, atol=0, equal_nan=True)


def test_subsampled_control_points_rejected(tmp_path):
    path = tmp_path / 'subsampled.nc'
    _write_l1c(path, control_points=True, cp_stride=2)
    with pytest.raises(ProductError, match='control points'):
        Level1_NASA(path)


# ------------------------------------------------------------------ loud errors

def test_missing_band_variable(tmp_path):
    path = tmp_path / 'noband.nc'
    _write_l1c(path, drop_band=412)
    with pytest.raises(ProductError, match='rhot_412'):
        Level1_NASA(path)


def test_missing_instrument(tmp_path):
    path = tmp_path / 'noinstr.nc'
    _write_l1c(path, instrument='')
    with pytest.raises(ProductError, match='instrument'):
        Level1_NASA(path)


def test_missing_platform(tmp_path):
    path = tmp_path / 'noplat.nc'
    _write_l1c(path, platform='')
    with pytest.raises(ProductError, match='platform'):
        Level1_NASA(path)


def test_unparseable_time_coverage(tmp_path):
    path = tmp_path / 'badtime.nc'
    _write_l1c(path, tstart='not-a-date')
    with pytest.raises(ProductError, match='time_coverage'):
        Level1_NASA(path)


def test_band_table_shorter_than_nref(tmp_path):
    path = tmp_path / 'shorttable.nc'
    _write_l1c(path, table_size=2)
    with pytest.raises(ProductError, match='band table'):
        Level1_NASA(path)


# ------------------------------------------------------------- other sensors

def test_viirs_attrs_and_shortname_override(tmp_path):
    path = tmp_path / 'N2024010v01.L1C.nc'
    _write_l1c(path, instrument='VIIRS', platform='SNPP', short_name='OCN_L1B_VIIRS')
    ds = Level1_NASA(path)
    assert ds.attrs[str(names.sensor)] == 'VIIRS'
    assert ds.attrs[str(names.shortname)] == 'OCN_L1B_VIIRS'
    assert ds.attrs['_srf_getter_arg'] == 'snpp_viirs'


# ---------------------------------------------------------- download utilities

def test_check_nasa_download(tmp_path):
    good = tmp_path / 'ok.nc'
    good.write_bytes(b'\x89HDF\r\n\x1a\n' + b'0' * 100)
    anc_nasa.check_nasa_download(good)              # no exception

    for head in (b'<!DOCTYPE html>', b'404 Error', b'403 Error'):
        bad = tmp_path / 'bad.html'
        bad.write_bytes(head + b' ' * 100)
        with pytest.raises(RuntimeError, match='EarthData'):
            anc_nasa.check_nasa_download(bad)


def test_nasa_download_url_routing(tmp_path, monkeypatch):
    calls = []

    def fake_download_url(url, dirname, **kwargs):
        calls.append((url, str(dirname), kwargs))
        return Path(dirname) / Path(url).name

    monkeypatch.setattr(anc_nasa, 'download_url', fake_download_url)

    # plain product name -> getfile
    anc_nasa.nasa_download('A2005005002500.L1C.nc', tmp_path)
    url, dirname, kwargs = calls[-1]
    assert url == 'https://oceandata.sci.gsfc.nasa.gov/getfile/A2005005002500.L1C.nc'
    assert dirname == str(tmp_path)
    assert kwargs['check_function'] is anc_nasa.check_nasa_download
    assert '--load-cookies ~/.urs_cookies' in kwargs['wget_opts']
    assert kwargs['lock_timeout'] == 3600
    assert kwargs['if_exists'] == 'skip'

    # S3* product -> sentinel getfile, .zip appended
    anc_nasa.nasa_download('S3A_OL_1_EFR_20240101', tmp_path)
    assert calls[-1][0] == ('https://oceandata.sci.gsfc.nasa.gov/sentinel/'
                            'getfile/S3A_OL_1_EFR_20240101.zip')

    # full URL passthrough
    anc_nasa.nasa_download('https://example.com/data/x.nc', tmp_path)
    assert calls[-1][0] == 'https://example.com/data/x.nc'

    # extra wget options are appended
    anc_nasa.nasa_download('A2005005002500.L1C.nc', tmp_path, wget_extra='-t 3')
    assert '-t 3' in calls[-1][2]['wget_opts']


def test_nasa_search(monkeypatch):
    def fake_check_output(cmd, shell=False):
        assert cmd.startswith('wget -q --post-data="')
        assert 'sensor=modis' in cmd
        assert 'sdate=2005-02-05' in cmd
        assert 'addurl=0' in cmd and 'results_as_file=1' in cmd
        return b'A2005005002500.L1C.nc B2005005002600.L1C.nc\n'

    monkeypatch.setattr(anc_nasa.subprocess, 'check_output', fake_check_output)
    out = anc_nasa.nasa_search(sensor='modis', sdate='2005-02-05')
    assert out == ['A2005005002500.L1C.nc', 'B2005005002600.L1C.nc']
