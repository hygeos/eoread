from datetime import date
from pathlib import Path

import numpy as np
import pytest

import tempfile

from eoread.ancillary.era5 import ERA5

from tempfile import TemporaryDirectory

def test_get():
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        era5 = ERA5(model=ERA5.models.reanalysis_single_level,
            directory=Path(tmpdir))
        ds = era5.get(variables=['mcc', 'tco3'], d=date(2019, 11, 30)) # download dataset
    
        variables = list(ds) # get dataset variables as list of str
        
        assert 'mcc'  not in variables
        assert 'tco3' not in variables
        assert 'mid_cloud_cover'    in variables
        assert 'total_column_ozone' in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0


def test_get_local_var_def_file():
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                    directory=Path(tmpdir), 
                    nomenclature_file=Path('tests/ancillary/inputs/nomenclature/variables.csv'))
        ds = era5.get(variables=['tco3'], d=date(2019, 11, 30)) # download dataset
    
        variables = list(ds) # get dataset variables as list of str
        
        assert 'tco3' not in variables
        assert 'local_total_column_ozone' in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0


def test_get_pressure_levels():
    
    # o3, q, t
    # ozone_mass_mixing_ratio', 'specific_humidity', 'temperature',
    
    with TemporaryDirectory() as tmpdir:

        era5 = ERA5(model=ERA5.models.reanalysis_pressure_levels,
                    directory=Path(tmpdir))
                    
        area = [1, -1, -1,  1]
        ds = era5.get(variables=['o3', 'q', 't'], d=date(2013, 11, 30), area=area) # download dataset
        
        variables = list(ds)
        
        assert 'o3' not in variables
        assert 'q'  not in variables
        assert 't'  not in variables
        assert 'ozone_mass_mixing_ratio' in variables
        assert 'specific_humidity' in variables
        assert 'temperature' in variables
        


def test_get_no_std():
    
    with TemporaryDirectory() as tmpdir:
        
        era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                    directory=Path(tmpdir), no_std=True)
        ds = era5.get(variables=['mcc', 'tco3'],d=date(2022, 11, 30)) # download dataset
        
        variables = list(ds)
        assert 'mcc'  in variables
        assert 'tco3' in variables
        assert 'mid_cloud_cover'    not in variables
        assert 'total_column_ozone' not in variables


def test_fail_get_offline():
    
    era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                directory=Path('tests/ancillary/inputs/ERA5'), offline=True)
    
    with pytest.raises(ResourceWarning):
        era5.get(variables=['mcc'],d=date(2001, 11, 30))
        
        
# downloading offline an already locally existing product should work
def test_download_offline():
    
    # empy file but with correct nomenclature
    era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                directory=Path('tests/ancillary/inputs/ERA5'), offline=True)

    f = era5.download(variables=['mcc', 'tco3'],d=date(2022, 11, 30))
    
    assert isinstance(f, Path)
        
                
def test_fail_download_offline():
    
    # empy file but with correct nomenclature
    era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                directory=Path('tests/ancillary/inputs/ERA5'), offline=True)

    with pytest.raises(ResourceWarning):
        era5.get(variables=['mcc', 'tco3'],d=date(2001, 11, 30)) 


# should fail because the specified local folder doesn't exists
def test_fail_folder_do_not_exist():
    
    with pytest.raises(FileNotFoundError):
        ERA5(model=ERA5.models.reanalysis_single_level,
             directory=Path('DATA/WRONG/PATH/TO/NOWHERE/')) 
        
        
def test_fail_non_defined_var():
    era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                directory=Path('tests/ancillary/inputs/ERA5'))
    
    with pytest.raises(KeyError):
        era5.get(variables=['non_existing_var'], d=date(2013, 11, 23))
    
    