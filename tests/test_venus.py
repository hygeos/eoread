#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import xarray as xr

import pytest


def Level1_VENUS(product: Path) -> xr.Dataset:
    # Will be moved to eoread/venus.py
    raise NotImplementedError


@pytest.mark.parametrize('product', [
    Path('/archive2/data/VENUS/VENUS-XS_20230116-112657-000_L1C_VILAINE_C_V3-1/'),
])
def test_venus(product: Path):
    """
    Test the implementation of VENUS Level1 reader

    https://venus.cnes.fr/en/VENUS/index.htm
    https://www.eoportal.org/satellite-missions/venus#eop-quick-facts-section
    https://www.theia-land.fr/product/venus-2/

    See S2 reader for examples with geotiff
    Sample data:
        /archive2/data/VENUS/
        Par exemple celle du 16 janvier 2023:
        Reflectance TOA: https://theia.cnes.fr/atdistrib/rocket/#/collections/VENUSVM05/547c9331-68e4-52d3-b854-ce938153928e
        Reflectance surface: https://theia.cnes.fr/atdistrib/rocket/#/collections/VENUSVM05/17c1ccc8-70b6-5658-b2b4-681d391c4406


    Objectives:
        - top of atmosphere reflectances or radiances
        - observation angles (sza, vza, raa)
        - latitude, longitude
        - SRFs
    """
    raise NotImplementedError