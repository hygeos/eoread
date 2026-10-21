import xarray as xr
import pytest

from ..conftest import get_sample
from eoread.legacy.modis import Level1_MODIS
from core.env import getdir
from .. import generic


@pytest.fixture(scope="module")
def level1_modis(): return get_sample('MODIS_L1_AQUA')

@pytest.fixture(params=[200, (200, 300)])
def chunks(request):
    return request.param

@pytest.fixture(params=[500, None])
def resolution(request):
    return request.param

@pytest.fixture()
def product_modis(level1_modis):
    return Level1_MODIS(level1_modis, resolution=True)


def test_l1c_instantiation(chunks, level1_modis, resolution):
    Level1_MODIS(level1_modis, chunks=chunks, resolution=resolution)
    
def test_l1c_main(product_modis):
    generic.Test.main(product_modis, angle_data=False)
    
def test_l1c_time(chunks, level1_modis): 
    params = {'filepath': level1_modis, 'chunks': chunks}
    generic.Test.execution_time(Level1_MODIS, params)

def test_l1c_subset(product_modis):
    generic.Test.subset(product_modis)
    
def test_l1c_lazy_load(product_modis):
    generic.Test.lazy_load(product_modis)
    
def test_plot(request, level1_modis):
    l1 = Level1_MODIS(level1_modis, resolution=1000)
    generic.plot(request, l1, '1.0', poi={"x": 1000, "y": 1000})
    
def test_flag_reader(product_modis):
    generic.Test.flagreader(product_modis)