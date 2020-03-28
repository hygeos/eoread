#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pytest
from tests.products import products as p

from eoread import eo
from eoread.meris import Level1_MERIS

from . import generic
from .generic import indices, param


@pytest.mark.parametrize('product', [p['prod_meris_L1_20060822']])
@pytest.mark.parametrize('split', [True, False])
def test_instantiation(product, split):
    Level1_MERIS(product['path'], split=split)


@pytest.mark.parametrize('product', [p['prod_meris_L1_20060822']])
def test_main(product):
    l1 = Level1_MERIS(product['path'])
    eo.init_Rtoa(l1)
    generic.test_main(l1)


@pytest.mark.parametrize('product', [p['prod_meris_L1_20060822']])
@pytest.mark.parametrize('chunks', [500, (300, 500)])
def test_read(product, param, indices, chunks):
    l1 = Level1_MERIS(product['path'], chunks=chunks)
    eo.init_Rtoa(l1)
    generic.test_read(l1, param, indices)


@pytest.mark.parametrize('product', [p['prod_meris_L1_20060822']])
def test_subset(product):
    l1 = Level1_MERIS(product['path'])
    generic.test_subset(l1)
