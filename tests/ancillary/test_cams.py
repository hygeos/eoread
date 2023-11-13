from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from eoread.ancillary.cams import CAMS
import pytest

from tempfile import TemporaryDirectory


def test_get():
    
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(directory=tmpdir)
        ds = cams.get(product='GACF', variables=['aod469', 'aod670'], d=date(2019, 3, 22))
    
        # check that the variables have been correctly renamed
        variables = list(ds)
        assert 'aod670' not in variables
        assert 'aod469' not in variables
        assert 'total_aerosol_optical_thickness_469nm' in variables
        assert 'total_aerosol_optical_thickness_670nm' in variables
        
        # check that the constructed variable has been computed
        assert 'total_aerosol_optical_thickness_550nm' in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0
    

def test_get_local_var_def_file():
    
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(directory=tmpdir, nomenclature_file='tests/ancillary/inputs/nomenclature/variables.csv')
        ds = cams.get(product='GACF', variables=['gtco3'], d=date(2019, 3, 22))
    
        # check that the variables have been correctly renamed
        variables = list(ds)
        assert 'gtco3' not in variables
        
        # check that the constructed variable has been computed
        assert 'local_total_column_ozone' in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0


def test_get_no_std():
    
    with TemporaryDirectory() as tmpdir:
        
        cams = CAMS(directory=tmpdir, no_std=True)
        ds = cams.get(product='GACF', variables=['aod469', 'aod670'], d=date(2019, 3, 22))
        
        # check that the variables have not changed and kept their original short names
        variables = list(ds)
        assert 'aod670' in variables
        assert 'aod469' in variables
        assert 'total_aerosol_optical_thickness_469nm' not in variables
        assert 'total_aerosol_optical_thickness_670nm' not in variables
        
        # check that the constructed variable has not been computed
        assert 'total_aerosol_optical_thickness_550nm' not in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0


def test_fail_get_offline():

    cams = CAMS(directory='tests/ancillary/inputs/CAMS/', offline=True)

    with pytest.raises(ResourceWarning) as excinfo:
        cams.get(product='GACF', variables=['gtco3', 'bcaod550'], d=date(2003, 3, 22))


def test_convert_units():
    
    with TemporaryDirectory() as tmpdir:
        
        cams = CAMS(directory=tmpdir)
        ds = cams.get(product='GACF', variables=['gtco3'], d=date(2019, 3, 22))
        
        # check that the variables have been correctly renamed
        variables = list(ds)
        
        assert 'gtco3' not in variables
        
        assert 'total_column_ozone' in variables
        
        assert ds['total_column_ozone'].attrs['units'] == 'Dobsons'

# downloading offline an already locally existing product should work
def test_download_offline():
    
    # empy file but with correct nomenclature
    cams = CAMS(directory='tests/ancillary/inputs/CAMS/', offline=True)

    f = cams.download(product='GACF', variables=['gtco3'], d=date(2009, 3, 22))
    
    assert isinstance(f, Path)
    

# downloading offline a not already locally present product should fail
def test_fail_download_offline():
    
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(directory=tmpdir, offline=True)

        with pytest.raises(ResourceWarning) as excinfo:
            cams.download(product='GACF', variables=['gtco3', 'bcaod550'], d=date(2003, 3, 22))
    

def test_fail_bad_product():
    
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(directory=tmpdir)
        
        with pytest.raises(ValueError) as excinfo:
            cams.get(product='ObviouslyWrongProduct', variables=['gtco3'], d=date(2013, 11, 23))


# should fail because the specified local folder doesn't exists
def test_fail_folder_do_not_exist():
    
    with pytest.raises(FileNotFoundError) as excinfo:
        CAMS(directory='DATA/WRONG/PATH/TO/NOWHERE/')
        

def test_fail_non_defined_var():
    
    with TemporaryDirectory() as tmpdir:
        cams = CAMS(directory=tmpdir)
        var = 'non_existing_var'
        
        with pytest.raises(KeyError) as excinfo:
            ds = cams.get(product='GACF', variables=[var], d=date(2013, 11, 23))
    
    