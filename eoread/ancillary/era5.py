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

from typing import Callable
from .era5_models import ERA5_Models
from eoread.ancillary.baseprovider import BaseProvider

from eoread.static import interface

class ERA5(BaseProvider):
    '''
    Ancillary data provider using ERA5

    - model: valid ERA5 models are listed in ERA5.models object
    - directory: local folder path, where to download files 
    - nomenclature_file: local file path to a nomenclature CSV to be used by the nomenclature module
    - no_std: bypass the standardization to the nomenclature module, keeping the dataset as provided

    '''
    
    models = ERA5_Models
    
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
    
    
    def __init__(self, model: Callable, directory: Path, nomenclature_file=None, offline: bool=False, verbose: bool=True, no_std: bool=False):
        
        name = 'ERA5'
        # call superclass constructor 
        BaseProvider.__init__(self, name=name, model=model, directory=directory, nomenclature_file=nomenclature_file, 
                              offline=offline, verbose=verbose, no_std=no_std)
        
        self.client = None # cdsapi 

        # ERA5 Reanalysis nomenclature (ads name: short name, etc..)
        era5_csv_file = Path(__file__).parent / 'era5.csv' # file path relative to the module
        self.model_specs = pd.read_csv(Path(era5_csv_file).resolve(), skipinitialspace=True)               # read csv file
        self.model_specs = self.model_specs.apply(lambda x: x.str.strip() if x.dtype == 'object' else x) # remove trailing whitespaces
        self.model_specs = self.model_specs[~self.model_specs['name'].astype(str).str.startswith('#')]                      # remove comment lines
        
        # General variable nomenclature preparation
        self.names = Nomenclature(provider=name, csv_file=nomenclature_file)

    @interface
    def download(self, variables: list[str], d: date, area: list=[90, -180, -90, 180]) -> Path:
        """
        Download ERA5 model for the given date
        
        - variables: list of strings of the CAMS variables short names to download ex: ['gtco3', 'aod550', 'parcs']
        - d: date of the data (not datetime)
        - area: [90, -180, -90, 180] → [north, west, south, east]
        """

        file_path = None
        
        # prepare variable attributes
        if not self.no_std:
            for var in variables: # verify var nomenclature has been defined in csv, beforehand
                self.names.assert_var_is_defined(var)                
                
        self.cds_variables = [self.get_cds_name(var) for var in variables] # get cds name equivalent from short name
        
        # verify beforehand that the var has been properly defined
        for var in variables: 
            self.names.assert_var_is_defined(var)
            if var not in list(self.model_specs['short_name'].values):
                raise KeyError(f'Could not find short_name {var} in csv file')
        
        
        # transform function name to extract only the acronym
        acronym = ''.join([i[0] for i in self.model.__name__.upper().split('_')])
        # ex: reanalysis_single_level → 'ESL'            

        # call download method
        file_path = self.directory / Path(self._get_filename(variables, d, acronym, area))    # get path
        
        if not file_path.exists():  # download if not already present
            if self.offline:        # download needed but deactivated → raise error
                raise ResourceWarning(f'Could not find local file {file_path}, offline mode is set')
            
            if self.verbose: print(f'downloading: {file_path.name}')
            self.model(self, file_path, d, area) # download file
            
        elif self.verbose: # elif → file already exists
            print(f'found locally: {file_path.name}')
                
        return file_path
        

    def get_cds_name(self, short_name):
        """
        Returns the variable's ADS name (used to querry the Atmospheric Data Store)
        """
        return self.model_specs[self.model_specs['short_name'] == short_name]['cds_name'].values[0]
    