#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import zipfile
import pytest
from tests import products as p
from tests.products import meris_product, sample_data_path
from eoread.meris import Level1_MERIS
from eoread import eo


@pytest.mark.parametrize('product', [p.prod_meris_L1_20060822])
def test_meris_file_available(meris_product):
    assert os.path.exists(meris_product), f'"{meris_product}" does not exist'


@pytest.mark.parametrize('product', [p.prod_meris_L1_20060822])
def test_instantiation(meris_product):
    Level1_MERIS(meris_product)


@pytest.mark.parametrize('product', [p.prod_meris_L1_20060822])
@pytest.mark.parametrize('param', ['latitude', 'longitude', 'sza', 'vza', 'Ltoa', 'Rtoa'])
def test_meris_1(meris_product, param):
    l1 = Level1_MERIS(meris_product)
    eo.init_Rtoa(l1)
    l1 = l1.isel(
        rows=slice(500, 550),
        columns=slice(500, 550),
    )
    l1[param].compute()
