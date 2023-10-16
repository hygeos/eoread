import pytest

from eoread.ancillary.nomenclature import Nomenclature
from pathlib import Path


def test_get_var():
    
    # test for MERRA2
    nm = Nomenclature(provider = 'MERRA2')
    
    assert nm.get_new_name('SPEED') == 'surface_wind_speed'
    assert nm.get_new_name('TQV')   == 'total_column_water_vapor'
    
    # test for CAMS
    nm_cams = Nomenclature(provider = 'CAMS')
    
    assert nm_cams.get_new_name('aod550')   == 'total_aerosol_optical_thickness_550nm'
    assert nm_cams.get_new_name('suaod550') == 'sulfate_aerosol_optical_thickness_550nm'


def test_fail_get_inexistant_var():
    
    # test for MERRA2
    nm = Nomenclature(provider = 'MERRA2')
    
    with pytest.raises(KeyError) as excinfo:
        nm.get_new_name('SPEEEEED')
        nm.get_new_name('NOTAVAR')