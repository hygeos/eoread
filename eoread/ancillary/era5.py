from eoread.fileutils import filegen
from .nomenclature import Nomenclature
from datetime import date
from pathlib import Path
from eoread import eo

# import ancillary.cdsapi_parser as cdsapi_parser
import xarray as xr
import pandas as pd
import numpy as np
import cdsapi



class ERA5:
    '''
    Ancillary data provider using ERA5

    - directory: local folder path, where to download files 

    '''
    
    def standardize(self, ds: xr.Dataset) -> xr.Dataset:
        '''
        Open an ERA5 file and format it for consistency
        with the other ancillary data sources
        '''
        
        # convert total_column_ozone to Dobsons
        # cf https://sacs.aeronomie.be/info/dobson.php
        if 'tco3' in ds:
            ds['tco3'] = (2.1415 * 10**-5) * ds['tco3'] #  kg.m^-2 -> Dobsons
            ds['tco3'].attrs['units'] = 'Dobsons'
        
        # if wind components, aggregate them as the mathematical norm
        if 'u10' in ds and 'v10' in ds:
            ds['horizontal_wind'] = np.sqrt(ds.u10**2 + ds.v10**2)
        
        ds = self.names.rename_dataset(ds) # rename dataset according to nomenclature module
        
        return ds
    
    
    def __init__(self, directory: Path, nomenclature_file=None, offline: bool=False, verbose: bool=True, no_std: bool=False):
        
        self.directory = Path(directory).resolve()
        if not self.directory.exists():
            raise FileNotFoundError(f'Directory "{self.directory}" does not exist. Use an existing directory.')
            
        self.offline = offline
        self.verbose = verbose
        self.no_std = no_std
        
        self.file_pattern = "ERA5_%s_%s_%s_%s.nc" # product, 'global'/'region', vars and date
        self.client = None

        # ERA5 Reanalysis nomenclature (ads name: short name, etc..)
        era5_csv_file = Path(__file__).parent / 'era5.csv' # file path relative to the module
        self.product_specs = pd.read_csv(Path(era5_csv_file).resolve(), skipinitialspace=True)               # read csv file
        self.product_specs = self.product_specs.apply(lambda x: x.str.strip() if x.dtype == 'object' else x) # remove trailing whitespaces
        self.product_specs = self.product_specs[~self.product_specs['name'].astype(str).str.startswith('#')]                      # remove comment lines
        
        # General variable nomenclature preparation
        self.names = Nomenclature(provider='ERA5', csv_file=nomenclature_file)


    def get(self, product:str, variables: list[str], d: date, area: list=[90, -180, -90, 180]) -> xr.Dataset:
        """
        Download and apply post-process to ERA5 reanalysis product for the given date
        
        - product: ERA5 string product 
        - variables: list of strings of the CAMS variables short names to download ex: ['gtco3', 'aod550', 'parcs']
        - d: date of the data (not datetime)
        - area: [90, -180, -90, 180] -> [top, left, bot, right]
        """
        
        filepath = self.download(product, variables, d, area)
                  
        ds = xr.open_mfdataset(filepath) # open dataset
        
        # correctly wrap longitudes if full area requested
        if area == [90, -180, -90, 180]:
            ds = eo.wrap(ds, 'longitude', -180, 180)
        
        if self.no_std:
            return ds
        return self.standardize(ds) # apply post process

    
    def download(self, product:str, variables: list[str], d: date, area: list=[90, -180, -90, 180]) -> Path:
        """
        Download ERA5 product for the given date
        
        - product: ERA5 string product 
        - variables: list of strings of the CAMS variables short names to download ex: ['gtco3', 'aod550', 'parcs']
        - d: date of the data (not datetime)
        - area: [90, -180, -90, 180] -> [top, left, bot, right]
        """
        
            # list of currently supported products
        products = [
            'reanalysis-era5-single-levels',
            'RASL',
            'reanalysis-era5-pressure-levels',
            'RAPL',
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
        self.cds_variables = [self.get_cds_name(var) for var in variables] # get cds name equivalent from short name
        
        # verify beforehand that the var has been properly defined
        for var in variables: 
            self.names.assert_var_is_defined(var)
            if var not in list(self.product_specs['short_name'].values):
                raise KeyError(f'Could not find short_name {var} in csv file')
        
        # find corresponding functions to product
        product_abrv = None
        downloader = None
        
        if product in ['RASL', 'reanalysis-era5-single-levels']:
            product_abrv = 'RASL'
            downloader = self._download_reanalysis_single_level
            
        elif product in ['RAPL', 'reanalysis-era5-pressure-levels']:
            product_abrv = 'RAPL'
            downloader = self._download_reanalysis_pressure_levels

            
        if downloader is None or product_abrv is None:
            raise ValueError(f"product '{product}' is not currently supported, \n currently supported products: \n{supported}")
        
        # call download method
        file_path = self.directory / Path(self._get_filename(d, product_abrv, area))    # get path
        
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
        

    def get_cds_name(self, short_name):
        """
        Returns the variable's ADS name (used to querry the Atmospheric Data Store)
        """
        return self.product_specs[self.product_specs['short_name'] == short_name]['cds_name'].values[0]


    @filegen(1)
    def _download_reanalysis_single_level(self, target, d, area):
        """
        Download a single file, containing 24 times, hourly resolution
        uses the CDS API. Uses a temporary file and avoid unnecessary download 
        if it is already present, thanks to fileutil.filegen 
        
        - target: path to the target file after download
        - d: date of the dataset
        """
        if self.client is None:
            self.client = cdsapi.Client()

        print(f'Downloading {target}...')
        self.client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': self.cds_variables,
                'year':[f'{d.year}'],
                'month':[f'{d.month:02}'],
                'day':[f'{d.day:02}'],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', 
                         '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', 
                         '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', ],
                'format':'netcdf',
                'area': area,
            },
            target)
    
    
    @filegen(1)
    def _download_reanalysis_pressure_levels(self, target, d, area):
        """
        Download a single file, containing 24 times, hourly resolution
        uses the CDS API. Uses a temporary file and avoid unnecessary download 
        if it is already present, thanks to fileutil.filegen 
        
        - target: path to the target file after download
        - d: date of the dataset
        """
        if self.client is None:
            self.client = cdsapi.Client()

        print(f'Downloading {target}...')
        self.client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'area': area,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'year':[f'{d.year}'],
                'month':[f'{d.month:02}'],
                'day':[f'{d.day:02}'],
                'pressure_level': [
                    '1', '2', '3',
                    '5', '7', '10',
                    '20', '30', '50',
                    '70', '100', '125',
                    '150', '175', '200',
                    '225', '250', '300',
                    '350', '400', '450',
                    '500', '550', '600',
                    '650', '700', '750',
                    '775', '800', '825',
                    '850', '875', '900',
                    '925', '950', '975',
                    '1000',
                ],
                'variable': self.cds_variables,
            },
            target
        )
    
            
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
    