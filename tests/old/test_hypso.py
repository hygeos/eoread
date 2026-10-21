from pathlib import Path
from ..conftest import get_sample
from eoread.legacy.hypso import Level1_HYPSO
import xarray as xr
from core.env import getdir

from .. import generic

import pytest 


@pytest.fixture(scope="module")
def level1_hypso() -> Path: return get_sample('DIR_SAMPLE_HYPSO')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request): return request.param

@pytest.fixture
def hypso_product(level1_hypso, chunks):
    return Level1_HYPSO(level1_hypso, chunks=chunks)


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_hypso, chunks):
    Level1_HYPSO(level1_hypso, chunks=chunks)

def test_main(hypso_product):
    generic.test_main(hypso_product, angle_data=True)
    
def test_time(level1_hypso, chunks): 
    params = {'filepath': level1_hypso, 'chunks': chunks}
    generic.test_execution_time(Level1_HYPSO, params)

def test_v1_compat(level1_hypso):
    v1_data = getdir("DIR_V1_COMPAT_DATA")
    l1 = Level1_HYPSO(level1_hypso, v1_compat=True)
    old = xr.open_dataset(v1_data/(level1_hypso.stem+'_res'))
    generic.compare_version(l1, old)
    
def test_lazy_load(hypso_product):
    generic.test_lazy_load(hypso_product)

@pytest.mark.skip()
@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(hypso_product, param, indices, scheduler):
    generic.test_read(hypso_product, param, indices, scheduler)

def test_subset(level1_hypso, chunks): 
    l1 = Level1_HYPSO(level1_hypso, chunks=chunks, metadata_template=[])
    generic.test_subset(l1)

def test_plot(request, hypso_product):
    generic.test_plot(request, hypso_product, 4)