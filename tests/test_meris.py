#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from matplotlib import pyplot as plt
from eoread.sample_products import product_getter
from eoread import eo
from eoread.meris import Level1_MERIS

from . import generic
from .generic import indices, param
from .conftest import savefig


product = pytest.fixture(params=[
    # 'prod_meris_L1_20060822',
    'prod_meris_L1_20080701',
])(product_getter)


@pytest.mark.parametrize('split', [True, False])
def test_instantiation(product, split):
    Level1_MERIS(product['path'], split=split)


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


def test_main(product):
    l1 = Level1_MERIS(product['path'])
    eo.init_Rtoa(l1)
    generic.test_main(l1)


@pytest.mark.parametrize('chunks', [500, (300, 500)])
@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
    # 'processes',
    # epr_api does not work with processes
    # (TypeError: no default __reduce__ due to non-trivial __cinit__)
])
def test_read(product, param, indices, chunks, scheduler):
    l1 = Level1_MERIS(product['path'], chunks=chunks)
    eo.init_Rtoa(l1)
    generic.test_read(l1, param, indices, scheduler)


def test_subset(product):
    l1 = Level1_MERIS(product['path'])
    generic.test_subset(l1)


def test_flag(product):
    """
    Check that flags are properly raised
    """
    l1 = Level1_MERIS(product['path'])
    assert (l1.flags > 0).any()

