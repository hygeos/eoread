#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from eoread.sample_products import get_sample_products

from eoread import eo
from eoread.reader.landsat8_oli import LATLON_GDAL, LATLON_NOGDAL, TOA_READ, Level1_L8_OLI

from . import generic

try:
    import gdal
except ModuleNotFoundError:
    gdal = None


@pytest.fixture()
def level1_landsat():
    return get_sample_products()['prod_oli_L1']['path']

@pytest.mark.parametrize('lat_or_lon', ['lat', 'lon'])
def test_latlon(level1_landsat, lat_or_lon):
    latlon_nogdal = LATLON_NOGDAL(level1_landsat, lat_or_lon)
    assert latlon_nogdal[:20, :10].shape == (20, 10)
    assert latlon_nogdal[:20:2, :10:2].shape == (10, 5)

    if gdal is not None:
        latlon_gdal = LATLON_GDAL(level1_landsat, lat_or_lon)
        assert latlon_gdal[:20, :10].shape == (20, 10)
        assert latlon_gdal[:20:2, :10:2].shape == (10, 5)

        assert latlon_gdal.shape == latlon_nogdal.shape
        np.testing.assert_allclose(
            latlon_gdal[::100, ::100],
            latlon_nogdal[::100, ::100])


@pytest.mark.parametrize('b', [440, 480, 560, 655, 865, 1375, 1610, 2200])
def test_toa_read(level1_landsat, b):
    r = TOA_READ(b, level1_landsat)
    assert r[5000:5050, 4000:4040].shape == (50, 40)
    assert r[5000:5050:2, 4000:4040:2].shape == (25, 20)


@pytest.mark.parametrize('split', [True, False])
def test_instantiate(level1_landsat, split):
    l1 = Level1_L8_OLI(level1_landsat, split=split)

    if split:
        assert 'Rtoa' not in l1
        assert 'Rtoa_440' in l1
    else:
        assert 'Rtoa' in l1
        assert 'Rtoa_440' not in l1


@pytest.mark.parametrize('angle', [True])
def test_main(level1_landsat, angle):
    l1 = Level1_L8_OLI(level1_landsat, angle_data=angle)
    generic.test_main(l1, angle)

@pytest.mark.parametrize('use_gdal', [False] if (gdal is None) else [True, False])
def test_read(param, indices, scheduler, use_gdal):
    l1 = Level1_L8_OLI(level1_landsat, use_gdal=use_gdal)
    eo.init_geometry(l1)
    generic.test_read(l1, param, indices, scheduler)

def test_subset(level1_landsat):
    l1 = Level1_L8_OLI(level1_landsat)
    generic.test_subset(l1)

@pytest.mark.parametrize('radio, angle', [
    ('radiance', False),
    ('reflectance', True)])
def test_radiometry(level1_landsat, radio, angle):
    l1 = Level1_L8_OLI(level1_landsat, radiometry=radio)
    generic.test_main(l1, angle)