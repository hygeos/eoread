#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pytest
import xarray as xr
from eoread.landsat8_oli import Level1_L8_OLI, LATLON, TOA_READ
from tests.custom_products import sample_landsat8_oli
from osgeo import gdal
import dask


@pytest.mark.parametrize('lat_or_lon', ['lat', 'lon'])
def test_latlon(lat_or_lon):
    latlon = LATLON(sample_landsat8_oli, lat_or_lon)
    assert latlon[:20, :10].shape == (20, 10)
    assert latlon[:20:2, :10:2].shape == (10, 5)


@pytest.mark.parametrize('i', range(1, 12))
def test_toa_read_1(i):
    filename = os.path.join(sample_landsat8_oli,
        f'LC08_L1TP_014028_20171002_20171014_01_T1_B{i}.TIF')
    dset = gdal.Open(filename)
    band = dset.GetRasterBand(1)
    print(f'Image dimensions are {band.XSize}x{band.YSize} ({filename})')
    xoff, yoff = 100, 50
    xs, ys = 70, 120
    data = band.ReadAsArray(
        xoff=xoff,
        yoff=yoff,
        win_xsize=xs,
        win_ysize=ys,
        )
    assert data.shape == (ys, xs)


@pytest.mark.parametrize('b', [440, 480, 560, 655, 865])
def test_toa_read_2(b):
    r = TOA_READ(b, sample_landsat8_oli)
    assert r[5000:5050, 4000:4040].shape == (50, 40)
    assert r[5000:5050:2, 4000:4040:2].shape == (25, 20)


@pytest.mark.parametrize('param', [
    'Rtoa_440',
    'Rtoa_560',
    'Rtoa_865',
    'sza',
    'vza',
    'saa',
    'vaa',
    'latitude',
    'longitude',
    ])
def test_landsat8_split(param):
    l1 = Level1_L8_OLI(sample_landsat8_oli, l8_angles='l8_angles/l8_angles')

    with dask.config.set(scheduler='single-threaded'):
        # l1[param][5000, 6000].compute()
        l1[param][:500, :400].compute()
        l1[param][6500:, 6500:].compute()
        l1[param][5000:5010, 4000:4020].compute()
        l1[param][4000:5000:100, 5000:6000:100].compute()
        l1[param][::100, ::100].compute()

        xr.testing.assert_allclose(
            l1[param][1000:1010, 500:510],
            l1.isel(rows=slice(1000, None),
                    columns=slice(500, None))[param][:10, :10])


def test_landsat8_merge():
    raise NotImplementedError