from eoread.ancillary.seaice import SeaIce
from datetime import date
import xarray as xr
import numpy as np

import pytest

from tempfile import TemporaryDirectory

from pathlib import Path
from datetime import datetime


def test_interp():
    
    
    with TemporaryDirectory() as tmpdir:
    
        # mode = "MY"
        mode = "NRT"
        
        seaice = SeaIce(directory=tmpdir, offline=False, mode=mode)
        ds_in = xr.open_mfdataset('/archive2/proj/PAR/SAMPLES/OUTPUT/S3A_OL_1_ERR____20210601T094955_20210601T103421_20210602T152714_2666_072_236______MAR_O_NT_002.SEN3.par.nc')

        ds_si = seaice.get(d=date(2023, 9, 30), lat=ds_in.Latitude, lon=ds_in.Longitude) 
                           
        ds_si.to_netcdf(f'/home/Joackim/tmp/interpolated_sea_ice_world_{mode}_REAL.nc')

        assert ds_in.dims['x'] == ds_si.dims['x']
        assert ds_in.dims['y'] == ds_si.dims['y']

# needs a valid .netrc file
def test_download_and_interpolate():
    
    
    with TemporaryDirectory() as tmpdir:
    
        seaice = SeaIce(directory=tmpdir, offline=False)
        ds_in = xr.open_mfdataset('tests/ancillary/inputs/sea_ice/thinned_input_L2.nc')
    
        ds_si = seaice.get(d=date(2023, 6, 23), lat=ds_in.Latitude, lon=ds_in.Longitude) 
        ds_si.to_netcdf(Path(tmpdir)/'interpolated_sea_ice.nc')

        assert ds_in.dims['x'] == ds_si.dims['x']
        assert ds_in.dims['y'] == ds_si.dims['y']
        

# should fail because no local file exist for this file
def test_fail_file_not_found_offline():
    
    with TemporaryDirectory() as tmpdir:
        seaice = SeaIce(directory=tmpdir, offline=True)
        
        dummy_lat = np.ones((40, 50)) # should fail before being used
        dummy_lon = np.ones((40, 50))
        
        with pytest.raises(FileNotFoundError) as excinfo:   
            seaice.get(d=date(1792, 7, 14), lat=dummy_lat, lon=dummy_lon)
    
    
# should fail because no files exists for this date
# needs a valid .netrc file
def test_fail_file_not_found_online():
    
    with TemporaryDirectory() as tmpdir:
    
        seaice = SeaIce(directory=tmpdir)
        
        dummy_lat = np.ones((40, 50)) # should fail before being used
        dummy_lon = np.ones((40, 50)) # so no need to load real values
        
        with pytest.raises(FileNotFoundError) as excinfo:
            seaice.get(d=date(1792, 7, 14), lat=dummy_lat, lon=dummy_lon)


# should fail  because the specified local folder doesn't exist
# same behavior as era5
def test_fail_folder_do_not_exist():
    
    with pytest.raises(FileNotFoundError) as excinfo:
        seaice = SeaIce(directory='DATA/WRONG/PATH/TO/NOWHERE/')
