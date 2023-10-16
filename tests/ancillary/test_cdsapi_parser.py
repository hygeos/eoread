from pathlib import Path

from eoread.ancillary import cdsapi_parser as pa

import pytest

def test_parse_correct_config():
    
    pa.read_multi_config('tests/ancillary/inputs/cdsapi_parser/correct.cfg')

def test_error_bad_syntax():
    
    with pytest.raises(SyntaxError) as excinfo:
        pa.read_multi_config('tests/ancillary/inputs/cdsapi_parser/bad_syntax.cfg')
        
def test_warning_too_many_keys():
    
    with pytest.raises(SyntaxWarning) as excinfo:
        pa.read_multi_config('tests/ancillary/inputs/cdsapi_parser/too_many_keys.cfg')
        