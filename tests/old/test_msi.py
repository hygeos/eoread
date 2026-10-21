#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from .. import generic
from ..conftest import get_sample
from eoread.legacy.msi import Level1_MSI, Level2_MSI


@pytest.fixture(scope="session")
def level1_msi(): 
    return get_sample('MSI_L1C')

@pytest.fixture(scope="session")
def level2_msi(): 
    return get_sample('MSI_L2A')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture(params=[None, 10])
def resolution(request):
    return request.param

@pytest.fixture
def S2_product(level1_msi, chunks, resolution):
    return Level1_MSI(level1_msi, chunks=chunks, resolution=resolution)


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_msi, resolution, chunks):
    Level1_MSI(level1_msi, chunks=chunks, resolution=resolution)

def test_main(level1_msi, chunks):
    l1 = Level1_MSI(level1_msi, chunks=chunks, resolution=60).compute()
    generic.Test.main(l1, angle_data=True)
    
def test_time(level1_msi, chunks): 
    params = {'dirname': level1_msi, 'chunks': chunks}
    generic.Test.execution_time(Level1_MSI, params)
    
def test_lazy_load(S2_product):
    generic.Test.lazy_load(S2_product)

@pytest.mark.skip()
@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(S2_product, param, indices, scheduler):
    eo.init_geometry(S2_product)
    generic.Test.read(S2_product, param, indices, scheduler)

def test_subset(level1_msi, chunks): 
    l1 = Level1_MSI(level1_msi, chunks=chunks, metadata_template=[])
    generic.Test.subset(l1)

def test_plot(request, level1_msi):
    l1 = Level1_MSI(level1_msi, resolution=60)
    generic.plot(request, l1, 'B4', poi={"x": 1000, "y": 1000})
    
def test_flag_reader(S2_product):
    generic.Test.flagreader(S2_product)

def test_l2_instantiation(level2_msi, chunks):
    Level2_MSI(level2_msi, chunks=chunks, resolution=60)