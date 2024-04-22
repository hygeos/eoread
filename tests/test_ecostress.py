import pytest

from eoread.reader.ecostress import Level1_ECOSTRESS, get_sample
from . import generic


@pytest.fixture()
def level1_ecostress():
    return get_sample()

@pytest.fixture()
def product_ecostress(level1_ecostress):
    return Level1_ECOSTRESS(level1_ecostress)

@pytest.mark.parametrize('radiometry',['radiance','reflectance'])
def test_radiometry(level1_ecostress, radiometry):
    l1 = Level1_ECOSTRESS(level1_ecostress, radiometry=radiometry)
    
    if radiometry == 'radiance':
        assert 'BT' not in l1 and 'Ltoa_tir' in l1
    else:                        
        assert 'BT' in l1 and 'Ltoa_tir' not in l1
        
@pytest.mark.parametrize('split',[True,False])
def test_split(level1_ecostress, split):
    l1 = Level1_ECOSTRESS(level1_ecostress, split=split)
    
    if split:
        assert 'BT' not in l1 and 'BT_1' in l1
    else:                        
        assert 'BT' in l1 and 'BT_1' not in l1

def test_main(product_ecostress):
    generic.test_main(product_ecostress)

def test_subset(product_ecostress):
    generic.test_subset(product_ecostress)