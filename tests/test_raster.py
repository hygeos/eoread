#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import rasterio
import xarray as xr
import numpy as np
from eoread.raster import ArrayLike_GDAL, gdal


files = [
    'SAMPLE_DATA/LANDSAT8_OLI/LC80140282017275LGN00/LC08_L1TP_014028_20171002_20171014_01_T1_B8.TIF',
    'SAMPLE_DATA/MSI/S2A_MSIL1C_20190419T105621_N0207_R094_T31UDS_20190419T130656.SAFE/GRANULE/L1C_T31UDS_A019968_20190419T110407/IMG_DATA/T31UDS_20190419T105621_B08.jp2',
]

@pytest.mark.parametrize('filename', files)
def test_read_rasterio(filename):
    rasterio.open(filename).read()

@pytest.mark.parametrize('filename', files)
def test_read_xarray(filename):
    xr.open_rasterio(filename).compute(scheduler='sync')

@pytest.mark.skipif(gdal is None, reason='GDAL is not installed')
@pytest.mark.parametrize('filename', files)
def test_gdal(filename):
    ArrayLike_GDAL(filename)[:, :]

@pytest.mark.skipif(gdal is None, reason='GDAL is not installed')
@pytest.mark.parametrize('filename', files)
def test_check(filename):
    """
    Check the consistency between different methods for reading
    """
    s1 = slice(10, 20)
    s2 = slice(100, 107)

    data0 = ArrayLike_GDAL(filename)[s1, s2]
    data1 = rasterio.open(filename).read(window=(s1, s2))
    data2 = xr.open_rasterio(filename).isel(band=0)[s1, s2].compute(scheduler='sync')

    assert np.allclose(data0, data1)
    assert np.allclose(data0, data2)
