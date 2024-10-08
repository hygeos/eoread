#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from eoread.sample_products import get_sample_products

from eoread import eo
from eoread.reader.landsat9_oli import LATLON_GDAL, LATLON_NOGDAL, TOA_READ, Level1_L9_OLI, get_sample

from . import generic
from .generic import indices, param, scheduler

try:
    import gdal
except ModuleNotFoundError:
    gdal = None


@pytest.fixture
def sample_landsat9_oli():
    return get_sample()

@pytest.mark.parametrize('lat_or_lon', ['lat', 'lon'])
def test_latlon(sample_landsat9_oli, lat_or_lon):
    latlon_nogdal = LATLON_NOGDAL(sample_landsat9_oli, lat_or_lon)
    assert latlon_nogdal[:20, :10].shape == (20, 10)
    assert latlon_nogdal[:20:2, :10:2].shape == (10, 5)

    if gdal is not None:
        latlon_gdal = LATLON_GDAL(sample_landsat9_oli, lat_or_lon)
        assert latlon_gdal[:20, :10].shape == (20, 10)
        assert latlon_gdal[:20:2, :10:2].shape == (10, 5)

        assert latlon_gdal.shape == latlon_nogdal.shape
        np.testing.assert_allclose(
            latlon_gdal[::100, ::100],
            latlon_nogdal[::100, ::100])


@pytest.mark.parametrize('b', [440, 480, 560, 655, 865, 1375, 1610, 2200])
def test_toa_read(sample_landsat9_oli, b):
    r = TOA_READ(b, sample_landsat9_oli)
    assert r[5000:5050, 4000:4040].shape == (50, 40)
    assert r[5000:5050:2, 4000:4040:2].shape == (25, 20)


@pytest.mark.parametrize('split', [True, False])
def test_instantiate(sample_landsat9_oli, split):
    l1 = Level1_L9_OLI(sample_landsat9_oli, split=split)

    if split:
        assert 'Rtoa' not in l1
        assert 'Rtoa_440' in l1
    else:
        assert 'Rtoa' in l1
        assert 'Rtoa_440' not in l1


def test_main():
    p = '/archive2/proj/QTIS_TRISHNA/L8L9/USA/LC09_L1TP_014034_20220618_20230411_02_T1'
    l1 = Level1_L9_OLI(p)
    generic.test_main(l1)

@pytest.mark.parametrize('use_gdal', [False] if (gdal is None) else [True, False])
def test_read(sample_landsat9_oli, param, indices, scheduler, use_gdal):
    l1 = Level1_L9_OLI(sample_landsat9_oli, use_gdal=use_gdal)
    eo.init_geometry(l1)
    generic.test_read(l1, param, indices, scheduler)

def test_subset(sample_landsat9_oli):
    l1 = Level1_L9_OLI(sample_landsat9_oli)
    generic.test_subset(l1)

@pytest.mark.parametrize('radio, angle', [
    ('radiance', False),
    ('reflectance', True)])
def test_radiometry(sample_landsat9_oli, radio, angle):
    l1 = Level1_L9_OLI(sample_landsat9_oli, radiometry=radio)
    generic.test_main(l1, angle)