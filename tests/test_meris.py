#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
from eoread.meris import Level1_MERIS, Level1_MERIS_SAFE, get_sample
import xarray as xr
from core.env import getdir, getvar
from . import generic


product = '/archive2/data/EOREAD_TESTDATA/MERIS/MER_RR__1PRACR20080701_014028_000026402070_00003_33123_0000.N1'

@pytest.fixture(scope="module")
def level1_meris(): return get_sample(1)

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



# MERIS 4th reprocessing (SAFE format)

@pytest.fixture(scope="module")
def level1_meris_safe():
    p = getvar('LEVEL1_MERIS_SAFE', default='')
    if p == '':
        pytest.skip('LEVEL1_MERIS_SAFE is not defined')
    return Path(p)

@pytest.fixture
def product_meris_safe(level1_meris_safe, chunks):
    return Level1_MERIS_SAFE(level1_meris_safe, chunks=chunks)

def test_safe_main(product_meris_safe):
    generic.Test.main(product_meris_safe, angle_data=False)

def test_safe_lazy_load(product_meris_safe):
    generic.Test.lazy_load(product_meris_safe)

def test_safe_subset(product_meris_safe):
    generic.Test.subset(product_meris_safe)

def test_safe_spectral(product_meris_safe):
    ds = product_meris_safe
    assert len(ds.bands) == 15
    assert ds.Ltoa.attrs['units'] == 'mW/(m^2*sr*nm)'
    sub = ds.isel(x=slice(500, 520), y=slice(500, 520))
    assert sub.F0.notnull().all()
    assert 1600 < float(sub.F0.sel(bands='M01').mean()) < 1900
    assert 405 < float(sub.wav.sel(bands='M01').mean()) < 420

def test_safe_flag_reader(product_meris_safe):
    generic.Test.flagreader(product_meris_safe)
