import pytest

from eoread.reader.modis import Level1_MODIS, get_sample
from . import generic


@pytest.fixture()
def level1_modis():
    return get_sample()

@pytest.fixture()
def product_modis(level1_modis):
    return Level1_MODIS(level1_modis)

@pytest.mark.parametrize('radiometry',['radiance','reflectance'])
def test_radiometry(level1_modis, radiometry):
    l1 = Level1_MODIS(level1_modis, radiometry=radiometry)
    
    if radiometry == 'radiance':
        assert 'BT' not in l1 and 'Ltoa_tir' in l1
    else:                        
        assert 'BT' in l1 and 'Ltoa_tir' not in l1
        
@pytest.mark.parametrize('split',[True,False])
def test_split(level1_modis, split):
    l1 = Level1_MODIS(level1_modis, split=split)
    
    if split:
        assert 'BT' not in l1 and 'BT_1' in l1
    else:                        
        assert 'BT' in l1 and 'BT_1' not in l1

def test_main(product_modis):
    generic.test_main(product_modis)

def test_subset(product_modis):
    generic.test_subset(product_modis)