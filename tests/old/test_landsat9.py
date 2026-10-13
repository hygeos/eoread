#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from .conftest import require_sample
from eoread.legacy.oli import Level1_OLI
from . import generic


product_l1 = pytest.fixture(lambda: require_sample('LEVEL1_L9'), scope='module')
product_l2 = pytest.fixture(lambda: require_sample('LEVEL2_L9'), scope='module')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture()
def product_l9_oli(product_l1, chunks):
    return Level1_OLI(product_l1, chunks=chunks)


def test_l1c_instantiate(product_l1):
    l1 = Level1_OLI(product_l1)

def test_l1c_main(product_l9_oli):
    generic.Test.main(product_l9_oli, angle_data=False)
    
def test_l1c_time(product_l1, chunks): 
    params = {'dirname': product_l1, 'chunks': chunks}
    generic.Test.execution_time(Level1_OLI, params)

def test_l1c_subset(product_l9_oli):
    generic.Test.subset(product_l9_oli)
    
def test_l1c_lazy_load(product_l9_oli):
    generic.Test.lazy_load(product_l9_oli)
    
def test_flag_reader(product_l9_oli):
    generic.Test.flagreader(product_l9_oli)

def test_plot(request, product_l1):
    l1 = Level1_OLI(product_l1)
    generic.plot(request, l1, '4', poi={"x": 3000, "y": 3000})