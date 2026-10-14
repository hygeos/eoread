import pytest

from .. import generic
from ..conftest import require_sample
from eoread.legacy.pace import Level1B_PACE_OCI
from core.tests import conftest



@pytest.fixture
def level1_oci():
    return require_sample('LEVEL1_PACE')

@pytest.fixture
def OCI_product(chunks, level1_oci):
    return Level1B_PACE_OCI(level1_oci, chunks=chunks)

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


def test_instantiation(level1_oci, chunks):
    Level1B_PACE_OCI(level1_oci, chunks=chunks)

def test_main(level1_oci, chunks):
    l1 = Level1B_PACE_OCI(level1_oci, chunks=chunks).compute()
    generic.Test.main(l1, angle_data=True)
    
def test_time(level1_oci, chunks): 
    params = {'product_pace_oci': level1_oci, 'chunks': chunks}
    generic.Test.execution_time(Level1B_PACE_OCI, params)
    
def test_lazy_load(OCI_product):
    generic.Test.lazy_load(OCI_product)

def test_subset(level1_oci, chunks): 
    l1 = Level1B_PACE_OCI(level1_oci, chunks=chunks, metadata_template=[])
    generic.Test.subset(l1)

def test_plot(request, level1_oci):
    l1 = Level1B_PACE_OCI(level1_oci)
    generic.plot(request, l1, '892', poi={"x": 1000, "y": 1000})
    
def test_flag_reader(OCI_product):
    generic.Test.flagreader(OCI_product)




