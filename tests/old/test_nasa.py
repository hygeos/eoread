#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from core.env import getvar

from eoread.legacy.make_L1C import makeL1C
from eoread.legacy.nasa import Level1_NASA
from tests import generic
from tests.conftest import get_sample

nasa_products_L1A = {
    "HAWKEYE": {
        # SEAHAWK1_HAWKEYE.20230701T160442.L1A.nc
        "var": "LEVEL1_HAWKEYE",
        "band_nir": 867,
        "poi": {"x": 100, "y": 5000},
    },
    "SEAWIFS": {
        # SEASTAR_SEAWIFS_GAC.20000312T030717.L1A.nc
        "var": "LEVEL1_SEAWIFS",
        "band_nir": 865,
        "poi": {"x": 50, "y": 750},
    },
    "MODIS": {
        # A2008106124500.L1A_LAC
        "var": "LEVEL1_MODISA",
        "band_nir": 859,
        "poi": {"x": 500, "y": 500},
    },
    "VIIRS": {
        # V2019086125400.L1A_SNPP.nc
        "var": "LEVEL1_VIIRS_SNPP",
        "band_nir": 862,
        "poi": {"x": 1750, "y": 2250},
    },

}


@pytest.fixture(params=nasa_products_L1A.values(), ids=list(nasa_products_L1A.keys()))
def product_L1A(request):
    prod = request.param
    get_sample(prod["var"])
    return prod


@pytest.mark.parametrize(
    "method",
    [
        "docker",
        "shell",
    ],
)
def test_L1C(method, product_L1A):
    with TemporaryDirectory() as tmpdir:
        p = Path(getvar(product_L1A["var"]))
        out = makeL1C(p, Path(tmpdir), method=method, eline=100)
        assert out.exists()
        print(f'Generated {out}')


def _check_latlon_order(ds):
    """
    Check that latitude increases from bottom to top and longitude increases
    from left to right in the dataset.

    This validates the standard cartographic convention where:
    - rows index 0 corresponds to the top (highest latitude)
    - rows index -1 corresponds to the bottom (lowest latitude)
    - columns index 0 corresponds to the left (lowest longitude)
    - columns index -1 corresponds to the right (highest longitude)
    """
    from core.geo.naming import names

    # Latitude should increase from bottom (last row) to top (first row)
    # i.e., lat at row 0 > lat at row -1
    lat_col = ds[str(names.lat)].isel({names.columns: 0}).values
    if lat_col[0] < lat_col[-1]:
        raise ValueError(
            f'Latitude decreases from top to bottom ({lat_col[0]:.2f} -> {lat_col[-1]:.2f}). '
            'Expected latitude to increase from bottom to top. '
            'Try setting normalize_orientation=False.'
        )

    # Longitude should increase from left (first column) to right (last column)
    lon_row = ds[str(names.lon)].isel({names.rows: 0}).values
    if lon_row[0] > lon_row[-1]:
        raise ValueError(
            f'Longitude decreases from left to right ({lon_row[0]:.2f} -> {lon_row[-1]:.2f}). '
            'Expected longitude to increase from left to right. '
            'Try setting normalize_orientation=False.'
        )


def test_instantiate(product_L1A: dict):
    p = Path(getvar(product_L1A["var"]))
    product_L1C = makeL1C(p)
    l1 = Level1_NASA(product_L1C)
    pass


def test_plot(request, product_L1A: dict):
    p = Path(getvar(product_L1A["var"]))
    product_L1C = makeL1C(p)
    l1 = Level1_NASA(product_L1C)
    _check_latlon_order(l1)
    generic.plot(request, l1, product_L1A["band_nir"], product_L1A.get("poi", None))
