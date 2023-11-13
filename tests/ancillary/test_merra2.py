
from eoread.ancillary.merra2 import MERRA2

from pathlib import Path
from datetime import date

import pytest

from tempfile import TemporaryDirectory

def test_get():
    """
    Test basic get, either from local file or download
    Test variable name nomenclature
    """
    
    with TemporaryDirectory() as tmpdir:
    
        merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                    directory=tmpdir,
                    )
        
        ds = merra.get(product='M2T1NXRAD', variables=['CLDTOT', 'TAUTOT'], d=date(2023, 9, 10))
        
        # check that the variables have been correctly renamed
        variables = list(ds)
        assert 'CLDTOT' not in variables
        assert 'TAUTOT' not in variables
        assert 'total_cloud_cover' in variables
        assert 'total_cloud_optical_thickness' in variables


def test_get_local_var_def_file():
    """
    Test basic get, either from local file or download
    Test variable name nomenclature
    """
    
    with TemporaryDirectory() as tmpdir:
    
        merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                    directory=tmpdir,
                    nomenclature_file='tests/ancillary/inputs/nomenclature/variables.csv'
                    )
        
        ds = merra.get(product='M2I1NXASM', variables=['TO3'], d=date(2023, 9, 10))
        
        # check that the variables have been correctly renamed
        variables = list(ds)
        assert 'TO3' not in variables
        assert 'local_total_column_ozone' in variables


def test_no_std():
    """
    Test basic get, either from local file or download
    Test variable name nomenclature
    """
    with TemporaryDirectory() as tmpdir:
    
        merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                    directory=tmpdir,
                    no_std=True,
                    )
        
        ds = merra.get(product='M2T1NXRAD', variables=['CLDTOT'], d=date(2023, 9, 10))
        
        # check that the variables have not changed and kept their original short names
        variables = list(ds)
        assert 'CLDTOT' in variables
        assert 'total_cloud_cover' not in variables


def test_fail_get_offline():
    
    with TemporaryDirectory() as tmpdir:
        
        merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                    directory=tmpdir,
                    offline=True,
                    )
        
        with pytest.raises(ResourceWarning) as excinfo:
            merra.get(product='M2T1NXRAD', variables=['TAUTOT', 'LWTUPCLRCLN'], d=date(2023, 9, 10))


def test_download_offline():
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
            directory='tests/ancillary/inputs/MERRA2/',
            offline=True,
            )
            
    f = merra.download(product='M2I1NXINT', variables=['TQV'], d=date(2023, 9, 10))

    assert isinstance(f, Path)
    
    
def test_fail_download_offline():
    merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
            directory='tests/ancillary/inputs/MERRA2/',
            offline=True,
            )
    
    with pytest.raises(ResourceWarning) as excinfo:
        merra.download(product='M2I1NXINT', variables=['TQV', 'TOX'], d=date(2023, 9, 10))


def test_merge_diff_products():
    """
    Test downloading from different merra2 products
    """
    with TemporaryDirectory() as tmpdir:
    
        merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                    directory=tmpdir,
                    )
                    
        ds = merra.get(product='M2T1NXRAD', variables=['CLDTOT', 'TAUTOT'], d=date(2023, 9, 10))
        ds = ds.merge(
            merra.get(product='M2I1NXINT', variables=['TQV'], d=date(2023, 9, 10))
        )
        
        # check that the variables have been correctly renamed
        variables = list(ds)
        assert 'TQV' not in variables
        
        assert 'total_column_water_vapor' in variables


def test_get_range():
    
    """
    Test basic get, either from local file or download
    Test variable name nomenclature
    """
    
    with TemporaryDirectory() as tmpdir:
        
        merra = MERRA2(config_file='tests/ancillary/inputs/merra2.json', 
                    directory=tmpdir,
                    )
        
        ds = merra.get_range(product='M2I1NXINT', variables=['TQV'], 
                                d1=date(2023, 9, 10), d2=date(2023, 9, 11)
                                )
                            
    # check that the merge happened over time dim
    assert len(ds.time.values) == 2 * 24 

    