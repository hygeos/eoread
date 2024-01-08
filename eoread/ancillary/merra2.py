from pydap.cas.urs import setup_session
import xarray as xr

from datetime import datetime, date, timedelta
from pathlib import Path
import requests

from eoread.fileutils import filegen
from eoread.cache import cache_json 
from eoread import download as dl

from .nomenclature import Nomenclature
from .merra2parser import Merra2Parser

from typing import Callable

# from typing import Callable
from .merra2_models import MERRA2_Models
from eoread.ancillary.baseprovider import BaseProvider

from eoread.static import interface

class MERRA2(BaseProvider):
    '''
    Ancillary data provider for MERRA-2 data from NASA's GES DISC
    https://uat.gesdisc.eosdis.nasa.gov/
    uses the OPeNDAP protocol, and credentials from .netrc file
    
    currently only supports single levels variables
    
    - model: valid MERRA2 models are listed in MERRA2.models object
             /!\\ currently only single levels are supported
    - directory: local folder path, where to download files 
    - config_file: path to the local file where to store the web-scraped MERRA2 config data
    - no_std: bypass the standardization to the nomenclature module, keeping the dataset as provided
    
    '''
    
    models = MERRA2_Models
    
    host = 'urs.earthdata.nasa.gov' # server to download from
    auth = dl.get_auth(host)        # credentials from netrc file
    base_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/' # base url for the OPeNDAP link
    # ex: https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXAER.5.12.4/2015/07/MERRA2_400.tavg1_2d_aer_Nx.20150705.nc4
    
    def standardize(self, ds):
        '''
        Modify a MERRA2 dataset variable names according to the name standard from nomenclature.py
        '''
            
        return self.names.rename_dataset(ds)
    
    @interface
    def __init__(self, model: Callable, directory: Path, nomenclature_file=None, offline:bool=False, 
                 verbose:bool=True, no_std: bool=False, config_file: Path=Path('merra2.json'),
                ):

        name = 'MERRA2'
        # call superclass constructor 
        BaseProvider.__init__(self, name=name, model=model, directory=directory, nomenclature_file=nomenclature_file, 
                              offline=offline, verbose=verbose, no_std=no_std)        
            
        self.model = model.__name__ # trick to allow autocompletion
        self.config_file = Path(config_file).resolve()
        
        dat = date(2022, 12, 12) # TODO change hardcoded value ?
        parser = Merra2Parser()
        self.config = cache_json(Path(self.config_file))(parser.get_model_specs)(dat)
        
        # get models version, and verify all models have the same
        v = None
        for model, cfgs in self.config.items():
            if cfgs['version'] != v:
                if v is not None: 
                    raise ValueError('Different version between different MERRA2 models, file nomenclature would be ambigous \n /!\\ THIS ERROR should not happen')
                v = cfgs['version']
        
        self.ouput_file_pattern = 'MERRA2_%s_%s_%s_%s.nc' # %model %version %vars %date 
        
        if 'user' not in self.auth: 
            raise KeyError(f'Missing key \'user\' for host {MERRA2.host} in .netrc file')
        if 'password' not in self.auth: 
            raise KeyError(f'Missing key \'password\' for host {MERRA2.host} in .netrc file')
    
        self.names = Nomenclature(provider='MERRA2', csv_file=nomenclature_file)
    
    @interface
    def download(self, variables: list[str], d: date, area: None|list=None) -> Path:
        """
        Download the model if necessary, returns the corresponding file Path
        
        - variables: list of strings of the MERRA2 vars to download (merra2 names) ex: ['TO3', 'TQV', 'SLP']
        - d: date of the data (not datetime)
        """
        
        if area is None:
            area = [90, -180, -90, 180]
        
        # TODO implement
        if area != [90, -180, -90, 180]:
            raise NotImplementedError(f'Parameter area not yet implemented, currently defaulting to full globe')
        
        cfg = self.config[self.model]
        
        if not self.no_std:
            variables = [self.names.get_shortname(var) for var in variables]
        
        for var in variables:
            if var not in cfg['variables']:
                raise KeyError(f'Could not find variable {var} in model {self.model}')
        
        acronym = self.model
        file_path = self.directory / Path(self._get_filename(variables, d, acronym, area)) # get file path
        
        
        if not file_path.exists(): 
            if self.offline:
                raise ResourceWarning(f'Could not find local file {file_path}, and online mode is off')
                
            if self.verbose:
                print(f'downloading: {file_path.name}')
            self._download_file(file_path, variables, d)
        elif self.verbose:
                print(f'found locally: {file_path.name}')
        
        return file_path

        
 
    def _assossiate_product(self, config, variables):
        '''
        Returns a list of tupple like: [(model_name, var_list), ..]
        list is ordered (descending) by number of needed variables contained per model
        models with 0 needed vars are removed
        '''
        
        # for every model associate the list of needed variables it contains
        res = {}
        for model in config:
            res[model] = []
        
        for var in variables:
            found = False
            for model, cfg in config.items(): 
                if var in cfg['variables']: 
                    res[model].append(var)
                    found = True
            if not found:
                raise ValueError(f'Could not find any model that contains variable \'{var}\'')
        
        # list model by number of variables contained, remove models with 0 needed variables
        r = []
        for k, v in res.items():
            if len(v) == 0: 
                continue # skip useless models
            r.append((k, v))
        
        r.sort(key = lambda item: len(item[1]), reverse=True)
        return r
    
        
    
    @filegen(1)
    def _download_file(self, target: Path, variables: list[str], d: date):
        '''
        Download a single file, contains a day of the correcsponding MERRA-2 product's data 
        uses OPeNDAP protocol. Uses a temporary file and avoid unnecessary download 
        if it is already present, thanks to fileutil.filegen 
        
        - target: path to the target file after download
        - product: string representing the merra2 product e.g 'M2T1NXSLV'
        - d: date of the dataset
        '''
        
        if isinstance(d, datetime): # TODO change
            d = d.date()
        
        
        # build file OPeNDAP url
        # 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXAER.5.12.4/2015/07/MERRA2_400.tavg1_2d_aer_Nx.20150705.nc4'
        filename = self.config[self.model]['generic_filename'] % d.strftime('%Y%m%d')
        version = self.config[self.model]['version']
        url = MERRA2.base_url + self.model + '.' + version + d.strftime('/%Y/%m/') + filename
        
        # Download file
        session = requests.Session()
        session = setup_session(self.auth['user'], self.auth['password'], check_url=url)

        store = xr.backends.PydapDataStore.open(url, session=session)
        ds = xr.open_dataset(store)
        
        ds = ds[variables] # trim dataset to only keep desired variables
        
        ds.to_netcdf(target)