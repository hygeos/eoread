
from eoread.ancillary.merra2 import MERRA2

from pathlib import Path
from datetime import date

import pytest

def test_get():
    """
    Test basic get, either from local file or download
    Test variable name nomenclature
    """
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                   directory='tests/ancillary/download',
                   )
    
    ds = merra.get(product='M2T1NXRAD', variables=['CLDTOT', 'TAUTOT'], d=date(2012, 12, 10))
    
    # check that the variables have been correctly renamed
    variables = list(ds)
    assert 'CLDTOT' not in variables
    assert 'TAUTOT' not in variables
    assert 'total_cloud_cover' in variables
    assert 'total_cloud_optical_thickness' in variables


def test_no_std():
    """
    Test basic get, either from local file or download
    Test variable name nomenclature
    """
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                   directory='tests/ancillary/download',
                   no_std=True,
                   )
    
    ds = merra.get(product='M2T1NXRAD', variables=['CLDTOT', 'TAUTOT'], d=date(2012, 12, 10))
    
     # check that the variables have not changed and kept their original short names
    variables = list(ds)
    assert 'CLDTOT' in variables
    assert 'TAUTOT' in variables
    assert 'total_cloud_cover' not in variables
    assert 'total_cloud_optical_thickness' not in variables


def test_fail_get_offline():
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                directory='tests/ancillary/download',
                offline=True,
                )
    
    with pytest.raises(ResourceWarning) as excinfo:
        merra.get(product='M2T1NXRAD', variables=['TAUTOT', 'LWTUPCLRCLN'], d=date(2003, 12, 10))


def test_download_offline():
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
            directory='tests/ancillary/inputs/MERRA2/',
            offline=True,
            )
            
    f = merra.download(product='M2I1NXINT', variables=['TQV'], d=date(2012, 12, 10))

    assert isinstance(f, Path)
    
    
def test_fail_download_offline():
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
            directory='tests/ancillary/inputs/MERRA2/',
            offline=True,
            )
    
    with pytest.raises(ResourceWarning) as excinfo:
        merra.download(product='M2I1NXINT', variables=['TQV', 'TOX'], d=date(2001, 12, 10))


def test_merge_diff_products():
    """
    Test downloading from different merra2 products
    """
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                directory='tests/ancillary/download',
                )
                
    ds = merra.get(product='M2T1NXRAD', variables=['CLDTOT', 'TAUTOT'], d=date(2012, 12, 10))
    ds = ds.merge(
        merra.get(product='M2I1NXINT', variables=['TQV'], d=date(2012, 12, 10))
    )
    
    # check that the variables have been correctly renamed
    variables = list(ds)
    assert 'TQV' not in variables
    
    assert 'total_column_water_vapor' in variables


def test_get_multiple():
    
    """
    Test basic get, either from local file or download
    Test variable name nomenclature
    """
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                   directory='tests/ancillary/download',
                   )
    
    ds = merra.get_multiple(product='M2I1NXINT', variables=['TQV'], 
                            d1=date(2012, 12, 10), d2=date(2012, 12, 12)
                            )
                            
    # check that the merge happened over time dim
    assert len(ds.time.values) == 3 * 24 

def test_download_product():
    """
    Test downloading function
    verify local file exists after download
    """
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                directory='tests/ancillary/download',
                )
    
    var = 'CLDTOT'
                
    target_file = Path(f'tests/ancillary/download/MERRA2_M2T1NXRAD_5.12.4_{var}_20121210.nc')
    target_file.resolve()
    
    if target_file.exists():
        target_file.unlink() # remove file if exists
    
    ds = merra.get(product='M2T1NXRAD', variables=[var], d=date(2012, 12, 10))
    
    assert target_file.exists() == True
    
    variables = list(ds)
    
    assert 'CLDTOT' not in variables
    assert 'total_cloud_cover' in variables
    