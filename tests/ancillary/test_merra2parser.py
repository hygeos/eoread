from eoread.ancillary.merra2parser import Merra2Parser

from eoread.cache import cache_json 

from datetime import date
from pathlib import Path
import pytest

def test_get_versions():
    
    parser = Merra2Parser()
    products = parser.get_products_vers()
    
    assert 'M2I1NXINT' in products
    assert 'M2T1NXRAD' in products
    
    assert products['M2I1NXINT'] == "5.12.4"
    assert products['M2T1NXRAD'] == "5.12.4"


def test_get_specs_local():
    
    parser = Merra2Parser()
    specs = cache_json(Path('merra2.json'))(parser.get_products_specs)(date(2012, 12, 10))
    
    assert 'M2I1NXINT' in specs
    assert 'M2T1NXRAD' in specs
    
    assert 'TQV'    in specs['M2I1NXINT']['variables']
    assert 'CLDTOT' in specs['M2T1NXRAD']['variables']
    
    