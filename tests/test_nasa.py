#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest

from eoread.nasa import Level1_NASA
from eoread.make_L1C import makeL1C
from tests.products import products as p
from . import generic
from .generic import indices, param


nasa_products = [
    'prod_A2008106_L1A_LAC',
    'prod_V2019086_L1A_SNPP',
    'prod_S2004115_L1A_GAC',
]

@pytest.mark.parametrize('pid', nasa_products)
def test_L1C(pid):
    filename = p[pid]['path']
    assert makeL1C(filename)

@pytest.mark.parametrize('pid', nasa_products)
def test_instantiate(pid):
    filename = p[pid]['path']
    Level1_NASA(makeL1C(filename))


@pytest.mark.parametrize('pid', nasa_products)
def test_main(pid):
    filename = p[pid]['path']
    l1 = Level1_NASA(makeL1C(filename))
    generic.test_main(l1)


@pytest.mark.parametrize('pid', nasa_products)
def test_read(pid):
    filename = p[pid]['path']
    l1 = Level1_NASA(makeL1C(filename))
    generic.test_read(l1, param, indices)


@pytest.mark.parametrize('pid', nasa_products)
def test_subset(pid):
    filename = p[pid]['path']
    l1 = Level1_NASA(makeL1C(filename))
    generic.test_subset(l1)
