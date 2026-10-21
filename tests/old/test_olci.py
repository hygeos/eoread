#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from ..conftest import get_sample
from eoread.legacy.olci import Level1_OLCI, Level2_OLCI
from .. import generic


olci_level1 = pytest.fixture(lambda: get_sample('OLCI_L1_EFR'), scope='module')
olci_level2 = pytest.fixture(lambda: get_sample('OLCI_L2_WFR'), scope='module')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def OLCI_product(chunks, olci_level1):
    return Level1_OLCI(olci_level1, chunks=chunks)

def test_plot(request, olci_level1):
    l1 = Level1_OLCI(olci_level1)
    generic.plot(request, l1, 'Oa09', poi = {"x": 1000, "y": 3000})
    
def test_l1c_instantiation(chunks, olci_level1):
    Level1_OLCI(olci_level1, chunks=chunks)

def test_plot(request, olci_level1):
    l1 = Level1_OLCI(olci_level1, v1_compat=True)
    generic.plot(request, l1, 865, poi = {"x": 1000, "y": 3000})
    
def test_l1c_main(OLCI_product):
    generic.Test.main(OLCI_product, angle_data=False)
    
def test_l1c_time(chunks, olci_level1): 
    params = {'dirname': olci_level1, 'chunks': chunks}
    generic.Test.execution_time(Level1_OLCI, params)

def test_l1c_subset(OLCI_product):
    generic.Test.subset(OLCI_product)
    
def test_l1c_lazy_load(OLCI_product):
    generic.Test.lazy_load(OLCI_product)
    
def test_flag_reader(OLCI_product):
    generic.Test.flagreader(OLCI_product)

def test_level2(chunks, olci_level2):
    Level2_OLCI(olci_level2, chunks=chunks)
