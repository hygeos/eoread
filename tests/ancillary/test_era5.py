from datetime import datetime, date
from pathlib import Path

import numpy as np
import pytest

import tempfile

from eoread.ancillary import ERA5

from tempfile import TemporaryDirectory

def test_get_datetime():
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        era5 = ERA5(model=ERA5.models.reanalysis_single_level,
            directory=Path(tmpdir))
        ds = era5.get(variables=['mid_cloud_cover', 'total_column_ozone'], dt=datetime(2019, 11, 30, 13, 35)) # download dataset
    
        variables = list(ds) # get dataset variables as list of str
        
        assert 'mcc'  not in variables
        assert 'tco3' not in variables
        assert 'mid_cloud_cover'    in variables
        assert 'total_column_ozone' in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0
        
        # check that the time interpolation occured
        assert len(np.atleast_1d(ds.time.values)) == 1


def test_get_computed():
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        era5 = ERA5(model=ERA5.models.reanalysis_single_level,
            directory=Path(tmpdir))
        ds = era5.get(variables=['surface_wind_speed'],
                      dt=datetime(2023, 3, 22, 14, 35))
                      
        # check that the variables have been correctly renamed
        variables = list(ds)

        # check that the constructed variable has been computed
        assert 'surface_wind_speed' in variables


def test_get_date():
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        era5 = ERA5(model=ERA5.models.reanalysis_single_level,
            directory=Path(tmpdir))
        ds = era5.get_day(variables=['mid_cloud_cover', 'total_column_ozone'], date=date(2019, 11, 30)) # download dataset
    
        variables = list(ds) # get dataset variables as list of str
        
        assert 'mcc'  not in variables
        assert 'tco3' not in variables
        assert 'mid_cloud_cover'    in variables
        assert 'total_column_ozone' in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0
        
        # check that the time interpolation did not occur
        assert len(np.atleast_1d(ds.time.values)) == 24


def test_get_local_var_def_file():
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                    directory=Path(tmpdir), 
                    nomenclature_file=Path('tests/ancillary/inputs/nomenclature/variables.csv'))
        ds = era5.get(variables=['local_total_column_ozone'], dt=datetime(2019, 11, 30, 13, 35)) # download dataset
    
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
        ds = era5.get(variables=['ozone_mass_mixing_ratio', 'specific_humidity', 'temperature'], 
                      dt=datetime(2013, 11, 30, 13, 35), area=area) # download dataset
        
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
        ds = era5.get(variables=['mcc', 'tco3'],dt=datetime(2022, 11, 30, 13, 35)) # download dataset
        
        variables = list(ds)
        assert 'mcc'  in variables
        assert 'tco3' in variables
        assert 'mid_cloud_cover'    not in variables
        assert 'total_column_ozone' not in variables


def test_fail_get_offline():
    
    era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                directory=Path('tests/ancillary/inputs/ERA5'), offline=True)
    
    with pytest.raises(ResourceWarning):
        era5.get(variables=['mid_cloud_cover'],dt=datetime(2001, 11, 30, 13, 35))
        
        
# downloading offline an already locally existing product should work
def test_download_offline():
    
    # empy file but with correct nomenclature
    era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                directory=Path('tests/ancillary/inputs/ERA5'), offline=True)

    f = era5.download(variables=['mid_cloud_cover', 'total_column_ozone'],d=date(2022, 11, 30))
    
    assert isinstance(f, Path)
        
                
def test_fail_download_offline():
    
    # empy file but with correct nomenclature
    era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                directory=Path('tests/ancillary/inputs/ERA5'), offline=True)

    with pytest.raises(ResourceWarning):
        era5.get(variables=['mid_cloud_cover', 'total_column_ozone'],dt=datetime(2001, 11, 30, 13, 35)) 


# should fail because the specified local folder doesn't exists
def test_fail_folder_do_not_exist():
    
    with pytest.raises(FileNotFoundError):
        ERA5(model=ERA5.models.reanalysis_single_level,
             directory=Path('DATA/WRONG/PATH/TO/NOWHERE/')) 
        
        
def test_fail_non_defined_var():
    era5 = ERA5(model=ERA5.models.reanalysis_single_level,
                directory=Path('tests/ancillary/inputs/ERA5'))
    
    with pytest.raises(LookupError):
        era5.get(variables=['non_existing_var'], dt=datetime(2013, 11, 23, 13, 35))
    
    