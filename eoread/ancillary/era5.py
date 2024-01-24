from ..utils.fileutils import filegen
from .nomenclature import Nomenclature
from datetime import date
from pathlib import Path
from ..utils.tools import wrap

# import ancillary.cdsapi_parser as cdsapi_parser
import xarray as xr
import pandas as pd
import numpy as np
import cdsapi

from typing import Callable
from .era5_models import ERA5_Models
from ..ancillary.baseprovider import BaseProvider

from ..utils.static import interface

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
        
        ds = self.names.rename_dataset(ds) # rename dataset according to nomenclature module
        
        if np.min(ds.longitude) == -180 and 175.0 <= np.max(ds.longitude) < 180:
            ds = wrap(ds, 'longitude', -180, 180)
        
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
        
        # computables variables and their requirements
        # functions needs to have the same parameters: (ds, new_var)
        self.computables['#windspeed'] = (ERA5.compute_windspeed, ['u10', 'v10'])
        
        
    # ----{ computed variables }----
    @staticmethod
    def compute_windspeed(ds, new_var) -> xr.Dataset:
        ds[new_var] = np.sqrt(ds.u10**2 + ds.v10**2)
        return ds
    # ------------------------------

    @interface
    def download(self, variables: list[str], d: date, area: None|list=None) -> Path:
        """
        Download ERA5 model for the given date
        
        - variables: list of strings of the CAMS variables short names to download ex: ['gtco3', 'aod550', 'parcs']
        - d: date of the data (not datetime)
        - area: [90, -180, -90, 180] → [north, west, south, east]
        """
        
        shortnames = variables # true if no_std is true

        # prepare variable attributes
        if self.no_std:
            for var in variables: # verify var nomenclature has been defined in csv, beforehand
                self.names.assert_shortname_is_defined(var)
            self.cds_variables = [self.get_cds_name(var) for var in variables] # get ads name equivalent from short name
        else:
            shortnames = [self.names.get_shortname(var) for var in variables]
            self.cds_variables = [self.get_cds_name(var) for var in shortnames] 
            
        # transform function name to extract only the acronym
        acronym = ''.join([i[0] for i in self.model.__name__.upper().split('_')])
        # ex: reanalysis_single_level → 'ESL'            

        # output file path
        file_path = self.directory / Path(self._get_filename(shortnames, d, acronym, area))    # get path
        
        if not file_path.exists():  # download if not already present
            if self.offline:        # download needed but deactivated → raise error
                raise ResourceWarning(f'Could not find local file {file_path}, offline mode is set')
            
            if self.verbose: 
                print(f'downloading: {file_path.name}')
            self.model(self, file_path, d, area) # download file
            
        elif self.verbose: # elif → file already exists
            print(f'found locally: {file_path.name}')
                
        return file_path
        

    def get_cds_name(self, short_name):
        """
        Returns the variable's ADS name (used to querry the Atmospheric Data Store)
        """
        # verify beforehand that the var has been properly defined
        if short_name not in list(self.model_specs['short_name'].values):
            raise KeyError(f'Could not find short_name {short_name} in csv file')
        
        return self.model_specs[self.model_specs['short_name'] == short_name]['cds_name'].values[0]
    