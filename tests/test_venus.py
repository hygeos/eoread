#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from eoread.legacy.venus import Level1_VENUS, Level2_VENUS
from .conftest import require_sample
from . import generic

import pytest


product_l1 = pytest.fixture(lambda: require_sample('LEVEL1_VENUS'), scope='session')
product_l2 = pytest.fixture(lambda: require_sample('LEVEL2_VENUS'), scope='session')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def VENUS_product(product_l1, chunks):
    return Level1_VENUS(product_l1, chunks=chunks)

   
def test_l1c_read_mask(product_l1):
    Level1_VENUS(product_l1, chunks=500, read_masks=True)
        
def test_l1c_instantiation(product_l1,chunks):
    Level1_VENUS(product_l1, chunks=chunks)
    
def test_l1c_main(VENUS_product):
    generic.Test.main(VENUS_product, angle_data=False)
    
def test_l1c_time(product_l1, chunks): 
    params = {'dirname': product_l1, 'chunks': chunks}
    generic.Test.execution_time(Level1_VENUS, params)

def test_l1c_subset(VENUS_product):
    generic.Test.subset(VENUS_product)
    
def test_l1c_lazy_load(VENUS_product):
    generic.Test.lazy_load(VENUS_product)

def test_level2(product_l2, chunks):
    Level2_VENUS(product_l2, chunks=chunks)

def test_plot(request, product_l1):
    l1 = Level1_VENUS(product_l1, chunks=500)
    generic.plot(request, l1, 'B4', poi = {"x": 5000, "y": 5000})
    
def test_flag_reader(product_l1):
    l1 = Level1_VENUS(product_l1, read_masks=True)
    generic.Test.flagreader(l1)