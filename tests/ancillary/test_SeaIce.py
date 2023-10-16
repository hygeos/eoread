import eoread.ancillary.SeaIce as si
from datetime import date
import xarray as xr
import numpy as np

import pytest

# needs a valid .netrc file
def test_download_and_interpolate():
    
    sicl = si.SeaIceLector(directory='tests/ancillary/download', offline=False)
    
    ds_in = xr.open_mfdataset('tests/ancillary/inputs/sea_ice/thinned_input_L2.nc')
    
    ds_si = sicl.get(d=date(2020, 6, 23), lat=ds_in.Latitude, lon=ds_in.Longitude) 
    ds_si.to_netcdf('tests/ancillary/inputs/interpolated_sea_ice.nc')

    assert ds_in.dims['x'] == ds_si.dims['x']
    assert ds_in.dims['y'] == ds_si.dims['y']
    

# should fail because no local file exist for this file
def test_fail_file_not_found_offline():
    
    sicl = si.SeaIceLector(directory='tests/ancillary/download', offline=True)
    
    dummy_lat = np.ones((40, 50)) # should fail before being used
    dummy_lon = np.ones((40, 50))
    
    with pytest.raises(FileNotFoundError) as excinfo:   
        sicl.get(d=date(1792, 7, 14), lat=dummy_lat, lon=dummy_lon)
    
    
# should fail because no files exists for this date
# needs a valid .netrc file
def test_fail_file_not_found_online():
    
    sicl = si.SeaIceLector(directory='tests/ancillary/download')
    
    dummy_lat = np.ones((40, 50)) # should fail before being used
    dummy_lon = np.ones((40, 50)) # so no need to load real values
    
    with pytest.raises(FileNotFoundError) as excinfo:
        sicl.get(d=date(1792, 7, 14), lat=dummy_lat, lon=dummy_lon)


# should fail  because the specified local folder doesn't exist
# same behavior as era5
def test_fail_folder_do_not_exist():
    
    with pytest.raises(FileNotFoundError) as excinfo:
        sicl = si.SeaIceLector(directory='DATA/WRONG/PATH/TO/NOWHERE/')
