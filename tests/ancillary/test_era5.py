
import pytest
from datetime import date

from pathlib import Path

from eoread.ancillary.era5 import ERA5

def test_get():
    
    era5 = ERA5(directory='tests/ancillary/download')

    ds = era5.get(product='RASL', variables=['mcc', 'tco3'], d=date(2022, 11, 30)) # download dataset
    
    variables = list(ds)
    
    assert 'mcc'  not in variables
    assert 'tco3' not in variables
    assert 'mid_cloud_cover'    in variables
    assert 'total_column_ozone' in variables


def test_get_no_std():
    
    era5 = ERA5(directory='tests/ancillary/download', no_std=True)
    ds = era5.get(product='RASL', variables=['mcc', 'tco3'],d=date(2022, 11, 30)) # download dataset
    
    variables = list(ds)
    assert 'mcc'  in variables
    assert 'tco3' in variables
    assert 'mid_cloud_cover'    not in variables
    assert 'total_column_ozone' not in variables


def test_fail_get_offline():
    
    era5 = ERA5(directory='tests/ancillary/inputs/ERA5', offline=True)
    
    with pytest.raises(ResourceWarning) as excinfo:
        era5.get(product='RASL', variables=['mcc'],d=date(2001, 11, 30))
        
        
# downloading offline an already locally existing product should work
def test_download_offline():
    
    # empy file but with correct nomenclature
    era5 = ERA5(directory='tests/ancillary/inputs/ERA5', offline=True)

    f = era5.download(product='RASL', variables=['mcc', 'tco3'],d=date(2022, 11, 30))
    
    assert isinstance(f, Path)
        
                
def test_fail_download_offline():
    
    # empy file but with correct nomenclature
    era5 = ERA5(directory='tests/ancillary/inputs/ERA5', offline=True)

    with pytest.raises(ResourceWarning) as excinfo:
        era5.get(product='RASL', variables=['mcc', 'tco3'],d=date(2001, 11, 30)) 


def test_fail_bad_product():
    era5 = ERA5(directory='tests/ancillary/inputs/ERA5')
    
    with pytest.raises(ValueError) as excinfo:
        era5.get(product='ObviouslyWrongProduct', variables=['gtco3'], d=date(2013, 11, 23))

# should fail because the specified local folder doesn't exists
def test_fail_folder_do_not_exist():
    
    with pytest.raises(FileNotFoundError) as excinfo:
        ERA5(directory='DATA/WRONG/PATH/TO/NOWHERE/') 
        
        
def test_fail_non_defined_var():
    era5 = ERA5(directory='tests/ancillary/inputs/ERA5')
    var = 'non_existing_var'
    
    with pytest.raises(KeyError) as excinfo:
        era5.get(product='RASL', variables=[var], d=date(2013, 11, 23))
    
    