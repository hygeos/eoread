#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from core.geo.naming import names
from eoread.olci import get_sample, Level1_OLCI, Level2_OLCI
from . import generic


olci_level1 = pytest.fixture(lambda: get_sample(1), scope='module')
olci_level2 = pytest.fixture(lambda: get_sample(2), scope='module')

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


def test_l1c_tie_points(OLCI_product):
    """
    Variables interpolated from the tie points (geometry and meteo) are float32,
    chunked like the radiances, and can be computed on a subset
    """
    ds = OLCI_product
    spatial_chunks = ds.Ltoa.chunks[1:]
    for var in ['sza', 'saa', 'vza', 'vaa', 'horizontal_wind',
                'sea_level_pressure', 'total_column_ozone']:
        assert ds[var].dtype == 'float32', var
        assert ds[var].chunks == spatial_chunks, var
    sub = ds.isel({str(names.rows): slice(1000, 1010),
                   str(names.columns): slice(2000, 2010)})
    assert sub.sza.notnull().all()
