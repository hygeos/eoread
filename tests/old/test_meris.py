#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
from ..conftest import get_sample
from eoread.legacy.meris import Level1_MERIS
import xarray as xr
from core.env import getdir
from .. import generic


product = '/archive2/data/EOREAD_TESTDATA/MERIS/MER_RR__1PRACR20080701_014028_000026402070_00003_33123_0000.N1'

@pytest.fixture(scope="module")
def level1_meris(): return get_sample('MERIS_L1')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def product_meris(level1_meris, chunks):
    return Level1_MERIS(level1_meris, chunks=chunks)


def test_instantiation(level1_meris, chunks):
    Level1_MERIS(level1_meris, chunks=chunks)

def test_main(product_meris):
    generic.Test.main(product_meris, angle_data=True)
    
def test_time(level1_meris, chunks): 
    params = {'filepath': level1_meris, 'chunks': chunks}
    generic.Test.execution_time(Level1_MERIS, params)
    
def test_lazy_load(product_meris):
    generic.Test.lazy_load(product_meris)
    
def test_subset(level1_meris, chunks): 
    l1 = Level1_MERIS(level1_meris, chunks=chunks, metadata_template=[])
    generic.Test.subset(l1)

def test_plot(request, product_meris):
    generic.plot(request, product_meris, '4')
    
def test_flag_reader(product_meris):
    generic.Test.flagreader(product_meris)

