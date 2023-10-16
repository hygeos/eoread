# -*- coding: utf-8 -*-


"""
Copernicus ancillary data provider for Sea-Ice-Concentration 
"""

from pathlib import Path
from datetime import date # no need for time interpolation, and therefore no need for datetime

from eoread import download as dl
from ftplib import FTP

import xarray as xr
import numpy as np
import pyproj


_ftp_configs = []
"""
Different ftp sources and paths to download files, severals can be specified and 
will be used as fallbacks if needed
"""

_data_src_1 = {
    'host':             'my.cmems-du.eu',
    'base_folder':      '/Core/SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009/',
    'sub_folder_north': 'OSISAF-GLO-SEAICE_CONC_CONT_TIMESERIES-NH-LA-OBS/%Y/%m/', 
    'sub_folder_south': 'OSISAF-GLO-SEAICE_CONC_CONT_TIMESERIES-SH-LA-OBS/%Y/%m/',
    'file_name_north':  'ice_conc_nh_ease2-250_cdr-v3p0_%Y%m%d1200.nc',
    'file_name_south':  'ice_conc_sh_ease2-250_cdr-v3p0_%Y%m%d1200.nc',
}
_ftp_configs.append(_data_src_1)
    
_data_src_2 = {
    'host':             'my.cmems-du.eu',
    'base_folder':      '/Core/SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009/',
    'sub_folder_north': 'OSISAF-GLO-SEAICE_CONC_TIMESERIES-NH-LA-OBS/%Y/%m/', 
    'sub_folder_south': 'OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS/%Y/%m/',
    'file_name_north':  'ice_conc_nh_ease2-250_cdr-v3p0_%Y%m%d1200.nc',
    'file_name_south':  'ice_conc_sh_ease2-250_cdr-v3p0_%Y%m%d1200.nc',
}
_ftp_configs.append(_data_src_2)


"""
Helper Functions
"""
def _get_ftp_paths_north(d: date, ftp_config: dict) -> Path: 
    """
    Static method that returns a Path object containing the ftp path and filename of the corresponding file, 
    depending on the server Context's nomenclature, for northern hemisphere
    """
    
    sub_folder = d.strftime(ftp_config['sub_folder_north'])
    file_name = d.strftime(ftp_config['file_name_north'])
    
    return Path(ftp_config['base_folder'] + sub_folder + file_name)


def _get_ftp_paths_south(d: date, ftp_config: dict) -> Path: 
    """
    tatic method that returns a Path object containing the ftp path and filename of the corresponding file, 
    depending on the server Context's nomenclature, for southern hemisphere
    """
    
    sub_folder = d.strftime(ftp_config['sub_folder_south'])
    file_name = d.strftime(ftp_config['file_name_south'])
    
    return Path(ftp_config['base_folder'] + sub_folder  + file_name)


def _get_ftp_connexion(host) -> FTP:
    """
    Static method that returns an FTP object with credentials from .netrc
    """
    auth = dl.get_auth(host)
    return FTP(host, auth['user'], auth['password'])
    

def _interp(fn_n: Path, fn_s: Path, lat: np.ndarray, lon: np.ndarray, 
            variables: list[str]=['ice_conc']) -> xr.DataArray:
    """
    Returns the Sea Ice Concentration data interpoated to the specified lat-lon
    Uses Pyproj to transform the lat-lon to polar coordinates then interpolate afterward  
    
    - fn_n: local filename for the northen hemisphere file
    - fn_s: local filename for the southern hemisphere file
    - lat, lon: coordinates on which to interpolate Sea Ice Conc
    """
    
    # will project the output's latlon to both north and south polar coordinates
    # so that we can use a linear interpolation for the data
    
    ds_n = xr.open_mfdataset(fn_n) # northern seaice dataset
    ds_s = xr.open_mfdataset(fn_s) # southern seaice dataset
    
    crs_n = pyproj.CRS('EPSG:6931') # northern polar coordinates system
    crs_s = pyproj.CRS('EPSG:6932') # southern polar coordinates system
    
    # northern Sea Ice
    p = pyproj.Proj(crs_n)
    x_n, y_n = p(longitude=lon, latitude=lat)
    
    x_n = x_n / 1000 # convert from meters to kilometers
    y_n = y_n / 1000
    
    x_n = xr.DataArray(dims=('yc', 'xc'), data=x_n) # add dimension information
    y_n = xr.DataArray(dims=('yc', 'xc'), data=y_n) # to allow interpolation
    
    ds_interp_n = ds_n[variables].interp(xc=x_n, yc=y_n) # interpolate ice_conc over projected latlon
    
    # southern Sea Ice
    p = pyproj.Proj(crs_s)
    x_s, y_s = p(longitude=lon, latitude=lat)
    
    x_s = x_s / 1000 # convert from meters to kilometers
    y_s = y_s / 1000
    
    x_s = xr.DataArray(dims=('yc', 'xc'), data=x_s) # add dimension information
    y_s = xr.DataArray(dims=('yc', 'xc'), data=y_s) # to allow interpolation
    
    ds_interp_s = ds_s[variables].interp(xc=x_s, yc=y_s) # interpolate ice_conc over projected latlon
    
    # combine north and south data
    combined = ds_interp_n.combine_first(ds_interp_s)
    
    # remove attributes
    for var in combined: combined[var].attrs = {}
    combined.attrs = {}
    lat.attrs = {}
    lon.attrs = {}
    
    return combined.rename({'yc': 'y', 'xc': 'x'}).assign_coords({'lat': lat, 'lon': lon})

class SeaIceLector:
    """
    Ancillary date provider using Copernicus
    https://data.marine.copernicus.eu/product/SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009/description
    """
    
    def __init__(self,
                 directory='ANCILLARY/SeaIce/',
                 ftp_configs=_ftp_configs,
                 offline=False,
                 verbose=False):
                 
        self.directory = Path(directory).resolve()
        self.ftp_configs = ftp_configs
        self.client = None
        self.offline = offline
        self.verbose = verbose
        
        assert isinstance(ftp_configs, list) and len(ftp_configs) >= 1
        
        if not self.directory.exists():
            raise FileNotFoundError(
                f'Directory "{self.directory}" does not exist. '
                'Please create it for hosting SeaIce files.')
    
    
    def get(self, d: date, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
        """
        Download source files and return the (interpolated) Sea Ice Concentration for the
        specified latlon tuple and date
        
        - d: date of the file requested
        - latlon: tuple of numpy arrays, used to interpolate the Sea Ice Concentration
        """
        
        assert isinstance(d, date)
        
        files_found = False
        ftp = None
        
        for cfg in self.ftp_configs:
            
            # check if files already exists at self.directory
            local_f_north = self.directory / Path(d.strftime(cfg['file_name_north']))
            local_f_south = self.directory / Path(d.strftime(cfg['file_name_south']))
            
            if local_f_south.is_file() and local_f_north.is_file(): files_found=True; break # skip download
            if self.offline: continue # skip download, but check other cfg for other nomenclatures 
            
            ftp_n = _get_ftp_paths_north(d, cfg) # file names on ftp server
            ftp_s = _get_ftp_paths_south(d, cfg)
            
            # create new ftp connexion if needed
            if not 'ftp' in cfg: 
                cfg['ftp'] = _get_ftp_connexion(cfg['host']); 
            ftp = cfg['ftp'] # get ftp connexion
            
            # check if both files exists
            ftp_n_check = dl.ftp_file_exists(ftp, ftp_n) # check if file north exists 
            ftp_s_check = dl.ftp_file_exists(ftp, ftp_s) # check if file south exists
            
            if not(ftp_n_check and ftp_s_check): continue
            
            if (ftp_n_check and ftp_s_check):
                try:
                    # args: (FTP, local_file_path,  ftp_folder_path)
                    dl.ftp_download(ftp, self.directory / ftp_n.name, ftp_n.parent)
                    dl.ftp_download(ftp, self.directory / ftp_s.name, ftp_s.parent)
                except: raise
                
                files_found = True
                break # do not attempt to download on other contexts
            
        if not files_found: raise FileNotFoundError(f'Could not find any corresponding file for date {d}')
        
        return _interp(
            fn_n = local_f_south, 
            fn_s = local_f_south, 
            lat = lat, 
            lon = lon)
    
    