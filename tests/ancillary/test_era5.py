from datetime import date
from pathlib import Path

import numpy as np
import pytest

import tempfile

from eoread.ancillary.era5 import ERA5

def test_get():
    ds = None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        era5 = ERA5(directory=tmpdir)
        ds = era5.get(product='RASL', variables=['mcc', 'tco3'], d=date(2022, 11, 30)) # download dataset
    
    assert ds is not None
    
    variables = list(ds) # get dataset variables as list of str
    
    assert 'mcc'  not in variables
    assert 'tco3' not in variables
    assert 'mid_cloud_cover'    in variables
    assert 'total_column_ozone' in variables
    
    # test wrap
    assert np.max(ds.longitude.values) == 180.0
    assert np.min(ds.longitude.values) == -180.0

# 
def test_get_pressure_levels():
    
    # o3, q, t
    # ozone_mass_mixing_ratio', 'specific_humidity', 'temperature',

    era5 = ERA5(directory='tests/ancillary/download')
    area = [1, -1, -1, 1,]
    ds = era5.get(product='RAPL', variables=['o3', 'q', 't'], d=date(2013, 11, 30), area=area) # download dataset
    
    variables = list(ds)
    
    assert 'o3' not in variables
    assert 'q'  not in variables
    assert 't'  not in variables
    assert 'ozone_mass_mixing_ratio' in variables
    assert 'specific_humidity' in variables
    assert 'temperature' in variables
    


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
    
    