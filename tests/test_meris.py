#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from matplotlib import pyplot as plt
from tests.products import products as p
from eoread import eo
from eoread.meris import Level1_MERIS

from . import generic
from .generic import indices, param
from .conftest import savefig

meris_products = [
    p['prod_meris_L1_20060822'],
    p['prod_meris_L1_20080701'],
]


@pytest.mark.parametrize('product', meris_products)
@pytest.mark.parametrize('split', [True, False])
def test_instantiation(product, split):
    Level1_MERIS(product['path'], split=split)


@pytest.mark.parametrize('product', meris_products)
def test_preview(product, param, request):
    """
    Generate browses of various products
    """
    plt.figure()
    l1 = Level1_MERIS(product['path'])
    eo.init_Rtoa(l1)
    if l1[param].ndim == 2:
        plt.imshow(l1[param][::10, ::10])
    elif l1[param].ndim == 3:
        plt.imshow(l1[param][-1, ::10, ::10])
    plt.colorbar()
    savefig(request)


@pytest.mark.parametrize('product', meris_products)
def test_main(product):
    l1 = Level1_MERIS(product['path'])
    eo.init_Rtoa(l1)
    generic.test_main(l1)


@pytest.mark.parametrize('product', meris_products)
@pytest.mark.parametrize('chunks', [500, (300, 500)])
def test_read(product, param, indices, chunks):
    l1 = Level1_MERIS(product['path'], chunks=chunks)
    eo.init_Rtoa(l1)
    generic.test_read(l1, param, indices)


@pytest.mark.parametrize('product', meris_products)
def test_subset(product):
    l1 = Level1_MERIS(product['path'])
    generic.test_subset(l1)


@pytest.mark.parametrize('product', meris_products)
def test_flag(product):
    """
    Check that flags are properly raised
    """
    l1 = Level1_MERIS(product['path'])
    assert (l1.flags > 0).any()
