from .conftest import require_sample
from eoread.legacy.autodetect import Level1, Level2
import pytest

level1 = pytest.fixture(lambda: require_sample('LEVEL1_MSI'), scope='module')
level2 = pytest.fixture(lambda: require_sample('LEVEL2_MSI'), scope='module')

def test_level1(level1):
    l1 = Level1(level1)

def test_level2(level2):
    l2 = Level2(level2)

def test_wrong_level(level1):
    with pytest.raises(AssertionError):
        Level2(level1)