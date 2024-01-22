from eoread.ancillary.seaice import SeaIce
from datetime import date
import xarray as xr
import numpy as np

import pytest

from tempfile import TemporaryDirectory

from pathlib import Path
from datetime import date, timedelta

# needs a valid .netrc file
def test_download_and_interpolate():
    
    
    with TemporaryDirectory() as tmpdir:
    
        seaice = SeaIce(directory=tmpdir, offline=False)
        ds_in = xr.open_mfdataset('tests/ancillary/inputs/sea_ice/thinned_input_L2.nc')
    
        date_nrt = date.today() - timedelta(days=15)
    
        ds_si = seaice.get(d=date_nrt, latitude=ds_in.Latitude, longitude=ds_in.Longitude) 
        ds_si.to_netcdf(Path(tmpdir)/'interpolated_sea_ice.nc')

        assert ds_in.dims['x'] == ds_si.dims['x']
        assert ds_in.dims['y'] == ds_si.dims['y']
        

# should fail because no local file exist for this file
def test_fail_file_not_found_offline():
    
    with TemporaryDirectory() as tmpdir:
        seaice = SeaIce(directory=tmpdir, offline=True)
        
        dummy_lat = np.ones((40, 50)) # should fail before being used
        dummy_lon = np.ones((40, 50))
        
        with pytest.raises(FileNotFoundError):   
            seaice.get(d=date(1792, 7, 14), latitude=dummy_lat, longitude=dummy_lon)
    
    
# should fail because no files exists for this date
# needs a valid .netrc file
def test_fail_file_not_found_online():
    
    with TemporaryDirectory() as tmpdir:
    
        seaice = SeaIce(directory=tmpdir)
        
        dummy_lat = np.ones((40, 50)) # should fail before being used
        dummy_lon = np.ones((40, 50)) # so no need to load real values
        
        with pytest.raises(FileNotFoundError):
            seaice.get(d=date(1792, 7, 14), latitude=dummy_lat, longitude=dummy_lon)


# should fail  because the specified local folder doesn't exist
# same behavior as era5
def test_fail_folder_do_not_exist():
    
    with pytest.raises(FileNotFoundError):
        SeaIce(directory='DATA/WRONG/PATH/TO/NOWHERE/')
