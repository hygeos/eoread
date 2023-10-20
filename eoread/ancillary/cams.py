from eoread.fileutils import filegen
from .nomenclature import Nomenclature
from datetime import date
from pathlib import Path
from eoread import eo

import eoread.ancillary.cdsapi_parser as cdsapi_parser
import xarray as xr
import pandas as pd
import numpy as np
import cdsapi
import os

import warnings

class CAMS:
    """
    Ancillary data provider for CAMS products

    - directory: local folder path, where to download files 
    - no_std: bypass the standardization to the nomenclature module, keeping the dataset as provided
    
    \n/!\ Currently only support Single-Levels
    """
    
    def standardize(self, ds: xr.Dataset) -> xr.Dataset:
        '''
        Open a CAMS file and standardize it for consistency
        with the other ancillary data sources
        '''
        
        # convert total_column_ozone to Dobsons
        # cf https://sacs.aeronomie.be/info/dobson.php
        if 'gtco3' in ds:
            ds['gtco3'] = (2.1415 * 10**-5) * ds['gtco3'] #  kg.m^-2 -> Dobsons
            ds['gtco3'].attrs['units'] = 'Dobsons'
        
        # if wind components, aggregate them as the mathematical norm
        if '10u' in ds and '10v' in ds:
            ds['surface_wind_speed'] = np.sqrt(ds.u10**2 + ds.v10**2)
        
        if 'aod469' in ds and 'aod670' in ds:
            ds['aod550'] = \
                - np.log(ds['aod469']/ds['aod670']) /  np.log(469.0/670.0)

        ds = self.names.rename_dataset(ds) # rename dataset according to nomenclature module
        return eo.wrap(ds, 'longitude', -180, 180)
    
    
    def __init__(self, directory: Path, offline: bool=False, verbose: bool=True, no_std: bool=False):
                
        self.directory = Path(directory).resolve()
        if not self.directory.exists():
            raise FileNotFoundError(f'Directory "{self.directory}" does not exist. Use an existing directory.')
            
        self.offline = offline
        self.verbose = verbose
        self.no_std = no_std
        
        self.file_pattern = "CAMS_%s_%s_%s_%s.nc" # product, 'global'/'region', vars, date
        self.client = None
        
        # CAMS nomenclature (ads name: short name, etc..)
        cams_csv_file = Path(__file__).parent / 'cams.csv' # file path relative to the module
        self.product_specs = pd.read_csv(Path(cams_csv_file).resolve(), skipinitialspace=True)               # read csv file
        self.product_specs = self.product_specs.apply(lambda x: x.str.strip() if x.dtype == 'object' else x) # remove trailing whitespaces
        self.product_specs = self.product_specs[~self.product_specs['name'].astype(str).str.startswith('#')] # remove comment lines

        # General variable nomenclature preparation
        self.names = Nomenclature(provider='CAMS')
                
        # get credentials from .cdsapirc file
        self._parse_cdsapirc()
        
    
    def get(self, product:str, variables: list[str], d: date, area: list=[90, -180, -90, 180]) -> xr.Dataset:
        """
        Download and apply post-process to CAMS global forecast product for the given date
        Standardize the dataset according to the nomenclature module
     
        product example:   
        https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts
        https://confluence.ecmwf.int/display/CKB/CAMS%3A+Global+atmospheric+composition+forecast+data+documentation#heading-Table1SinglelevelFastaccessparameterslastreviewedon02Aug2023
        
        - product: CAMS string product 
        - variables: list of strings of the CAMS variables short names to download ex: ['gtco3', 'aod550', 'parcs']
        - d: date of the data (not datetime)
        - area: [90, -180, -90, 180] -> [north, west, south, east]
        
        """
            
        filepath = self.download(product, variables, d, area)
        
        ds = xr.open_mfdataset(filepath)                      # open dataset
        
        # correctly wrap longitudes if full area requested
        if area == [90, -180, -90, 180]:
            ds = eo.wrap(ds, 'longitude', -180, 180)
        
        if self.no_std: # do not standardize, return as is
            return ds
        return self.standardize(ds) # standardize according to nomenclature file

        
    
    def download(self, product: str, variables: list[str], d: date, area: list=[90, -180, -90, 180])  -> Path:
        """
        Download CAMS product for the given date
        
        https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts
        https://confluence.ecmwf.int/display/CKB/CAMS%3A+Global+atmospheric+composition+forecast+data+documentation#heading-Table1SinglelevelFastaccessparameterslastreviewedon02Aug2023
        
        - product: CAMS string product 
        - variables: list of strings of the CAMS variables short names to download ex: ['gtco3', 'aod550', 'parcs']
        - d: date of the data (not datetime)
         - area: [90, -180, -90, 180] -> [top, left, bot, right]
        """
        
        # list of currently supported products
        products = [
            'global-atmospheric-composition-forecasts',
            'GACF',
        ]
        
        # list of currently supported products
        supported = '' # better error messages
        for p in products: supported += p + "\n"
        
        if product not in products: # error, invalid product
            raise ValueError(f"product '{product}' is not currently supported, \n currently supported products: \n{supported}")
        
        file_path = None
        
        # prepare variable attributes
        if not self.no_std:
            for var in variables: # verify var nomenclature has been defined in csv, beforehand
                self.names.assert_var_is_defined(var)
                
        self.variables = variables                                         # short names
        self.ads_variables = [self.get_ads_name(var) for var in variables] # get ads name equivalent from short name
        
        # verify beforehand that the var has been properly defined
        for var in variables: 
            self.names.assert_var_is_defined(var)
            if var not in list(self.product_specs['short_name'].values):
                raise KeyError(f'Could not find short_name {var} in csv file')
        
        # find corresponding functions to product
        product_abrv = None
        downloader = None
        
        if product in ['GACF', 'global-atmospheric-composition-forecasts']:
            product_abrv = "GACF" # <- 'cams-global-atmospheric-composition-forecasts'
            downloader = self._download_GACF_file
        
        if downloader is None or product_abrv is None:
            raise ValueError(f"product '{product}' is not currently supported, \n currently supported products: \n{supported}")
        
        # call download method
        file_path = self.directory / Path(self._get_filename(d, product_abrv, area)) # get file path
        
        if file_path.exists():
            if self.verbose:
                print(f'found locally: {file_path.name}')
        else:
            if not self.offline:
                downloader(file_path, d, area) # download if needed (uses filegen) 
            else: # cannot download
                raise ResourceWarning(f'Could not find local file {file_path}, offline mode is set')
                
            if self.verbose: 
                print(f'downloading: {file_path.name}')
                
        return file_path
    
    
    @filegen(1)
    def _download_GACF_file(self, target: Path, d: date, area):
        """
        Download a single file, containing 24 times, hourly resolution
        uses the CDS API. Uses a temporary file and avoid unnecessary download 
        if it is already present, thanks to fileutil.filegen 
        
        - target: path to the target file after download
        - d: date of the dataset
        """
        
        if self.client is None:
            self.client = cdsapi.Client(url=self.cdsapi_cfg['url'], 
                                        key=self.cdsapi_cfg['key'])
            
        self.client.retrieve(
            'cams-global-atmospheric-composition-forecasts',
            {
                'date': str(d)+'/'+str(d),
                'type': 'forecast',
                'format': 'netcdf',
                'variable': self.ads_variables,
                'time': ['00:00', '12:00'],
                'leadtime_hour': ['0', '1', '2', '3', '4', '5', 
                                  '6', '7', '8', '9', '10', '11'],
                'area': area,
            }, target)
            
            
    def get_ads_name(self, short_name):
        """
        Returns the variable's ADS name (used to querry the Atmospheric Data Store)
        """
        return self.product_specs[self.product_specs['short_name'] == short_name]['ads_name'].values[0]
    
    
    def _parse_cdsapirc(self):
        """
        after retrieval the function sets attributes cdsapi_url and cdsapi_key to
        pass as parameter in the Client constructor
        """
        # taken from ECMWF's cdsapi code
        dotrc = os.environ.get("CDSAPI_RC", os.path.expanduser("~/.cdsapirc"))
        config = cdsapi_parser.read_config('ads', dotrc) 
        # save the credentials as attributes
        self.cdsapi_cfg = config
    

    def _get_filename(self, d: date, product: str, area) -> str:
        """
        Constructs and return the target filename according to the nomenclature specified
        in the attribute 'filename_pattern'
        """
        
        area_str = "global"
        if area != [90, -180, -90, 180]:
            area_str = "region"
        
        # construct chain of variables short name 
        vars = list(self.variables) # sort alphabetically, so that variables order doesn't matter
        vars.sort()
        
        vars_str = vars.pop(0)  # get first element without delimiter
        for v in vars: vars_str += '_' + v # apply delimiter and names
            
        d = d.strftime('%Y%m%d')
        return self.file_pattern % (product, area_str, vars_str, d)