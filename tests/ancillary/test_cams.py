from datetime import datetime, date
from pathlib import Path

import numpy as np
import pytest

from eoread.ancillary import CAMS
from tempfile import TemporaryDirectory

def test_get_datetime():
    
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir))
        ds = cams.get(variables=['total_aerosol_optical_depth_469nm', 
                                 'total_aerosol_optical_depth_670nm'],
                      dt=datetime(2020, 3, 22, 14, 35))
    
        # check that the variables have been correctly renamed
        variables = list(ds)
        assert 'aod670' not in variables
        assert 'aod469' not in variables
        assert 'total_aerosol_optical_depth_469nm' in variables
        assert 'total_aerosol_optical_depth_670nm' in variables
        
        # check that the constructed variable has been computed
        assert 'total_aerosol_angstrom_coefficient_550nm' in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0
        
        # check that the time interpolation occured
        assert len(np.atleast_1d(ds.time.values)) == 1
        

def test_get_computed():
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir))
        ds = cams.get(variables=['surface_wind_speed', 
                                 'total_aerosol_angstrom_coefficient_550nm'],
                      dt=datetime(2023, 3, 22, 14, 35))
                      
        # check that the variables have been correctly renamed
        variables = list(ds)

        # check that the constructed variable has been computed
        assert 'total_aerosol_angstrom_coefficient_550nm' in variables
        assert 'surface_wind_speed' in variables
        

def test_get_date():
    
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir))
        ds = cams.get_day(variables=['total_aerosol_optical_depth_469nm', 
                                 'total_aerosol_optical_depth_670nm'],
                     date=date(2020, 3, 22))
    
        # check that the variables have been correctly renamed
        variables = list(ds)
        assert 'aod670' not in variables
        assert 'aod469' not in variables
        assert 'total_aerosol_optical_depth_469nm' in variables
        assert 'total_aerosol_optical_depth_670nm' in variables
        
        # check that the constructed variable has been computed
        assert 'total_aerosol_angstrom_coefficient_550nm' in variables
        
        # test wrap
        assert np.max(ds.longitude.values) == 180.0
        assert np.min(ds.longitude.values) == -180.0
        
        # check that the time interpolation did not occur
        assert len(np.atleast_1d(ds.time.values)) == 24


def test_get_range():

    with TemporaryDirectory() as tmpdir:
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir))
        ds = cams.get_range(variables=['total_aerosol_optical_depth_469nm', 'total_aerosol_optical_depth_670nm'],
                      date_start=date(2020, 3, 22), date_end=date(2020, 3, 23))
        
        assert len(ds.time == 48)


def test_get_local_var_def_file():
    
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir), 
                    nomenclature_file=Path('tests/ancillary/inputs/nomenclature/variables.csv'))
        ds = cams.get(variables=['local_total_column_ozone'], dt=datetime(2019, 3, 22, 13, 35))
    
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
        
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir), no_std=True)
        ds = cams.get(variables=['aod469', 'aod670'], dt=datetime(2019, 3, 22, 13, 35))
        
        # check that the variables have not changed and kept their original short names
        variables = list(ds)
        assert 'aod670' in variables
        assert 'aod469' in variables
        assert 'total_aerosol_optical_depth_469nm' not in variables
        assert 'total_aerosol_optical_depth_670nm' not in variables
        
        # check that the constructed variable has not been computed
        assert 'total_aerosol_optical_depth_550nm' not in variables
        

def test_fail_get_offline():

    cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                directory=Path('tests/ancillary/inputs/CAMS/'), offline=True)

    with pytest.raises(ResourceWarning):
        cams.get(variables=['total_column_ozone', 'black_carbon_aerosol_optical_depth_550nm'], 
                 dt=datetime(2003, 3, 22, 13, 35))


def test_convert_units():
    
    with TemporaryDirectory() as tmpdir:
        
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir))
        ds = cams.get(variables=['total_column_ozone'], dt=datetime(2019, 3, 22, 13, 35))
        
        # check that the variables have been correctly renamed
        variables = list(ds)
        
        assert 'gtco3' not in variables
        
        assert 'total_column_ozone' in variables
        
        assert ds['total_column_ozone'].attrs['units'] == 'Dobsons'

# downloading offline an already locally existing product should work
def test_download_offline():
    
    # empy file but with correct nomenclature
    cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                directory=Path('tests/ancillary/inputs/CAMS/'), offline=True)

    f = cams.download(variables=['total_column_ozone'], d=date(2009, 3, 22))
    
    assert isinstance(f, Path)
    

# downloading offline a not already locally present product should fail
def test_fail_download_offline():
    
    with TemporaryDirectory() as tmpdir:
    
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir), offline=True)

        with pytest.raises(ResourceWarning):
            cams.download(variables=['total_column_ozone', 'black_carbon_aerosol_optical_depth_550nm'], d=date(2003, 3, 22))
    
# should fail because the specified local folder doesn't exists
def test_fail_folder_do_not_exist():
    
    with pytest.raises(FileNotFoundError):
        CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
             directory=Path('DATA/WRONG/PATH/TO/NOWHERE/'))
        

def test_fail_non_defined_var():
    
    with TemporaryDirectory() as tmpdir:
        cams = CAMS(model=CAMS.models.global_atmospheric_composition_forecast,
                    directory=Path(tmpdir))
                    
        with pytest.raises(LookupError):
            cams.get(variables=[ 'non_existing_var'], dt=datetime(2013, 3, 22, 13, 35))
    
    