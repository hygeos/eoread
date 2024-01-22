#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from tempfile import TemporaryDirectory
import pytest
from eoread.nasa import Level1_NASA, nasa_download
from eoread.reader.make_L1C import makeL1C
from eoread.sample_products import product_getter, get_sample_products
from . import generic
from .generic import indices, param

p = get_sample_products()


nasa_products = [
    'prod_A2008106_L1A_LAC',
    'prod_V2019086_L1A_SNPP',
    'prod_S2004115_L1A_GAC',
]

product = pytest.fixture(params=nasa_products)(product_getter)


def test_L1C(product):
    filename = product['path']
    assert makeL1C(filename)


def test_instantiate(product):
    filename = product['path']
    Level1_NASA(makeL1C(filename))


@pytest.mark.parametrize('pid', nasa_products)
def test_main(pid):
    filename = p[pid]['path']
    l1 = Level1_NASA(makeL1C(filename))
    generic.test_main(l1)


@pytest.mark.parametrize('pid', nasa_products)
def test_read(pid, param, indices):
    filename = p[pid]['path']
    l1 = Level1_NASA(makeL1C(filename))
    generic.test_read(l1, param, indices, scheduler='sync')


@pytest.mark.parametrize('pid', nasa_products)
def test_subset(pid):
    filename = p[pid]['path']
    l1 = Level1_NASA(makeL1C(filename))
    generic.test_subset(l1)

