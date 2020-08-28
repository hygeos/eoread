#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import numpy as np
import dask
import pytest
import xarray as xr
from osgeo import gdal
from tests.products import products as p

from eoread import eo
from eoread.landsat8_oli import LATLON_GDAL, LATLON_NOGDAL, TOA_READ, Level1_L8_OLI

from . import generic
from .generic import indices, param, scheduler


sample_landsat8_oli = p['prod_oli_L1']['path']

@pytest.mark.parametrize('lat_or_lon', ['lat', 'lon'])
def test_latlon(lat_or_lon):
    latlon_gdal = LATLON_GDAL(sample_landsat8_oli, lat_or_lon)
    assert latlon_gdal[:20, :10].shape == (20, 10)
    assert latlon_gdal[:20:2, :10:2].shape == (10, 5)

    latlon_nogdal = LATLON_NOGDAL(sample_landsat8_oli, lat_or_lon)
    assert latlon_nogdal[:20, :10].shape == (20, 10)
    assert latlon_nogdal[:20:2, :10:2].shape == (10, 5)

    assert latlon_gdal.shape == latlon_nogdal.shape
    np.testing.assert_allclose(
        latlon_gdal[::100, ::100],
        latlon_nogdal[::100, ::100])


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


@pytest.mark.parametrize('b', [440, 480, 560, 655, 865, 1375, 1610, 2200])
def test_toa_read_2(b):
    r = TOA_READ(b, sample_landsat8_oli)
    assert r[5000:5050, 4000:4040].shape == (50, 40)
    assert r[5000:5050:2, 4000:4040:2].shape == (25, 20)


@pytest.mark.parametrize('split', [True, False])
def test_instantiate(split):
    l1 = Level1_L8_OLI(sample_landsat8_oli, split=split)

    if split:
        assert 'Rtoa' not in l1
        assert 'Rtoa_440' in l1
    else:
        assert 'Rtoa' in l1
        assert 'Rtoa_440' not in l1


def test_main():
    l1 = Level1_L8_OLI(sample_landsat8_oli)
    generic.test_main(l1)


def test_read(param, indices, scheduler):
    l1 = Level1_L8_OLI(sample_landsat8_oli)
    eo.init_geometry(l1)
    generic.test_read(l1, param, indices, scheduler)


def test_subset():
    l1 = Level1_L8_OLI(sample_landsat8_oli)
    generic.test_subset(l1)
