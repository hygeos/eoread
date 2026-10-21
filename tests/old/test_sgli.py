#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import xarray as xr

from ..conftest import get_sample
from eoread.legacy.sgli import Level1_SGLI
from core.env import getdir
from .. import generic



@pytest.fixture(scope="module")
def level1_sgli() -> Path: return get_sample('LEVEL1_SGLI')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def sgli_product(level1_sgli, chunks):
    return Level1_SGLI(level1_sgli, chunks=chunks)


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_sgli, chunks):
    Level1_SGLI(level1_sgli, chunks=chunks)

def test_main(sgli_product):
    generic.Test.main(sgli_product, angle_data=True)
    
def test_time(level1_sgli, chunks): 
    params = {'filepath': level1_sgli, 'chunks': chunks}
    generic.Test.execution_time(Level1_SGLI, params)
    
def test_lazy_load(sgli_product):
    generic.Test.lazy_load(sgli_product)

@pytest.mark.skip()
@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(sgli_product, param, indices, scheduler):
    generic.Test.read(sgli_product, param, indices, scheduler)

def test_subset(level1_sgli, chunks): 
    l1 = Level1_SGLI(level1_sgli, chunks=chunks, metadata_template=[])
    generic.Test.subset(l1)

def test_plot(request, level1_sgli):
    l1 = Level1_SGLI(level1_sgli)
    generic.plot(request, l1, 'VN04(S04)')
    
def test_flag_reader(sgli_product):
    generic.Test.flagreader(sgli_product)