import pytest
from ..conftest import require_sample
from eoread.legacy.ecostress import Level1C_ECOSTRESS
from .. import generic


@pytest.fixture(scope="session")
def level1C_ecostress(): return require_sample('LEVEL1_ECOSTRESS')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture()
def product_ecostress(level1C_ecostress, chunks):
    return Level1C_ECOSTRESS(level1C_ecostress, chunks=chunks)


################################################################################
# Tests for Level-1
################################################################################
    
def test_l1c_instantiation(level1C_ecostress, chunks):
    Level1C_ECOSTRESS(level1C_ecostress, chunks=chunks)
    
def test_l1c_main(product_ecostress):
    generic.Test.main(product_ecostress, angle_data=False)
    
def test_l1c_time(level1C_ecostress, chunks): 
    params = {'filepath': level1C_ecostress, 'chunks': chunks}
    generic.Test.execution_time(Level1C_ECOSTRESS, params)

def test_l1c_subset(product_ecostress):
    generic.Test.subset(product_ecostress)
    
def test_l1c_lazy_load(product_ecostress):
    generic.Test.lazy_load(product_ecostress)

@pytest.mark.skip(
    'legacy reader output is north-up (row 0 = high latitude) and fails the '
    'south-up orientation invariant; fixed in eoread.readers.ecostress '
    '(see tests/test_ecostress.py::test_l1c_latlon)'
)
def test_latlon(product_ecostress):
    generic.Test.latlon(product_ecostress)

def test_plot(request, level1C_ecostress):
    l1 = Level1C_ECOSTRESS(level1C_ecostress, chunks=500)
    generic.plot(request, l1, '4', poi = {"x": 1000, "y": 3000})
    
def test_flag_reader(product_ecostress):
    generic.Test.flagreader(product_ecostress)