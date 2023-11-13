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

class MERRA2:
    '''
    Ancillary data provider for MERRA-2 data from NASA's GES DISC
    https://uat.gesdisc.eosdis.nasa.gov/
    uses the OPeNDAP protocol, and credentials from .netrc file
    
    currently only supports single levels variables
    
    - directory: local folder path, where to download files 
    - config_file: path to the local file where to store the web-scraped MERRA2 config data
    - no_std: bypass the standardization to the nomenclature module, keeping the dataset as provided
    
    '''
    
    host = 'urs.earthdata.nasa.gov' # server to download from
    auth = dl.get_auth(host)        # credentials from netrc file
    base_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/' # base url for the OPeNDAP link
    # ex: https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXAER.5.12.4/2015/07/MERRA2_400.tavg1_2d_aer_Nx.20150705.nc4
    
    def standardize(self, ds):
        '''
        Modify a MERRA2 dataset variable names according to the name standard from nomenclature.py
        '''
            
        return self.names.rename_dataset(ds)
    
    def __init__(self,
                *,
                 directory: Path = 'ANCILLARY/MERRA-2',
                 nomenclature_file=None,
                 config_file: Path = 'merra2.json',
                 offline = False,
                 verbose = True,
                 no_std: bool=False,
                ):
        
        self.directory = Path(directory).resolve()
        self.config_file = Path(config_file).resolve()
        self.verbose = verbose
        self.offline = offline
        self.no_std  = no_std
        
        dat = date(2012, 12, 12) # TODO change hardcoded value ?
        parser = Merra2Parser()
        self.config = cache_json(Path(self.config_file))(parser.get_products_specs)(dat)
        
        # get product version, and verify all products have the same
        v = None
        for product, cfgs in self.config.items():
            if cfgs['version'] != v:
                if v is not None: 
                    raise ValueError('Different version between different MERRA2 products, file nomenclature would be ambigous')
                v = cfgs['version']
        
        self.ouput_file_pattern = 'MERRA2_%s_%s_%s_%s.nc' # %product %version %vars %date 
        
        if not self.directory.exists():
            raise FileNotFoundError(
                f'Directory "{self.directory}" does not exist. '
                'Please create it for hosting MERRA-2 files.')
        
        if 'user' not in self.auth: 
            raise KeyError(f'Missing key \'user\' for host {MERRA2.host} in .netrc file')
        if 'password' not in self.auth: 
            raise KeyError(f'Missing key \'password\' for host {MERRA2.host} in .netrc file')
    
        self.names = Nomenclature(provider='MERRA2', csv_file=nomenclature_file)
    
    
    def download(self, product: str, variables: list[str], d: date) -> Path:
        """
        Download the product if necessary, returns the corresponding file Path
        
        - product: string of the MERRA2 product from which to download the variables ex: 'M2I1NXASM'
        - variables: list of strings of the MERRA2 vars to download (merra2 names) ex: ['TO3', 'TQV', 'SLP']
        - d: date of the data (not datetime)

        """
        
        if product not in self.config: 
            raise KeyError(f'Could not find config for \'{product}\' in {self.config_file}')
            
        cfg = self.config[product]
        
        for var in variables:
            if var not in cfg['variables']:
                raise KeyError(f'Could not find variable {var} in product {product}')
        
        file = Path(self._get_dst_filename(product, variables, d))
        path = self.directory / file
        
        if path.exists(): 
            if self.verbose:
                print(f'downloading: {file}')
        else:
            if not self.offline:
                self._download_file(path, product, variables, d)
            else:
                raise ResourceWarning(f'Could not find local file {path}, and online mode is off')
            
            if self.verbose: 
                print(f'downloading: {file}')
        
        return path
                
                
    def get(self, product: str, variables: list[str], d: date) -> xr.Dataset:
        '''
        Returns the corresponding xarray Dataset object, with the correct variables
        Download the product if necessary (with only the requested variables)
        
        - product: string of the MERRA2 product from which to download the variables ex: 'M2I1NXASM'
        - variables: list of strings of the MERRA2 vars to download (merra2 names) ex: ['TO3', 'TQV', 'SLP']
        - d: date of the data (not datetime)
        - no_std: bypass the standardization to the nomenclature module, keeping the dataset as provided
        '''
        
        # download file if necessary
        filepath = self.download(product, variables, d)

        ds = xr.open_dataset(filepath)
        
        if self.no_std: # do not standardize 
            return ds
        return self.standardize(ds)
        
    
    def get_range(self, product: str, variables: list[str], d1: date, d2: date) -> xr.Dataset:
        """
        Download, or just load if possible the according product, merge different days
        between d1 and d2 both included into a single dataset object
        """
        
        dates = [d1 + timedelta(days=dt_day) for dt_day in range((d2 - d1).days + 1)]
 
        ds = self.get(product, variables, dates.pop(0))
        
        # merge the rest of the dates
        for d in dates:
            ds = ds.merge(self.get(product, variables, d))
        
        return ds
            
        
 
    def _assossiate_product(self, config, variables):
        '''
        Returns a list of tupple like: [(product_name, var_list), ..]
        list is ordered (descending) by number of needed variables contained per product
        products with 0 needed vars are removed
        '''
        
        # for every product associate the list of needed variables it contains
        res = {}
        for product in config:
            res[product] = []
        
        for var in variables:
            found = False
            for product, cfg in config.items(): 
                if var in cfg['variables']: 
                    res[product].append(var)
                    found = True
            if not found:
                raise ValueError(f'Could not find any product that contains variable \'{var}\'')
        
        # list product by number of variables contained, remove products with 0 needed variables
        r = []
        for k, v in res.items():
            if len(v) == 0: continue # skip useless products
            r.append((k, v))
        
        r.sort(key = lambda item: len(item[1]), reverse=True)
        return r
    
    
    def _get_dst_filename(self, product: str, variables: list[str], d: date) -> str:
        '''
        Constructs and return the target filename according to the nomenclature specified
        in the attribute 'filename_pattern'
        '''
        
        if not product in self.config: raise KeyError(f'Could not find config for \'{product}\''
                                                     +f' in {self.config_file}')
        if isinstance(d, datetime): d = d.date() # convert to date
        
        # construct chain of variables short name 
        vars_str = variables[0]  # get first element without delimiter
        for v in variables[1:]: vars_str += '_' + v # apply delimiter and names
        
        # date and version    
        dstr = d.strftime('%Y%m%d')
        version =  self.config[product]['version']
        
        return self.ouput_file_pattern % (product, version, vars_str, dstr)        
        
    
    @filegen(1)
    def _download_file(self, target: Path, product: str, variables: list[str], d: date):
        '''
        Download a single file, contains a day of the correcsponding MERRA-2 product's data 
        uses OPeNDAP protocol. Uses a temporary file and avoid unnecessary download 
        if it is already present, thanks to fileutil.filegen 
        
        - target: path to the target file after download
        - product: string representing the merra2 product e.g 'M2T1NXSLV'
        - d: date of the dataset
        '''
        
        if not product in self.config: raise KeyError(f'Could not find config for \'{product}\''
                                                      +f' in {self.config_file}')
        if isinstance(d, datetime): d = d.date()
        
        
        # build file OPeNDAP url
        # 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXAER.5.12.4/2015/07/MERRA2_400.tavg1_2d_aer_Nx.20150705.nc4'
        filename = self.config[product]['generic_filename'] % d.strftime('%Y%m%d')
        version = self.config[product]['version']
        url = MERRA2.base_url + product + '.' + version + d.strftime('/%Y/%m/') + filename
        
        # Download file
        session = requests.Session()
        session = setup_session(self.auth['user'], self.auth['password'], check_url=url)

        store = xr.backends.PydapDataStore.open(url, session=session)
        ds = xr.open_dataset(store)
        
        ds = ds[variables] # trim dataset to only keep desired variables
        
        ds.to_netcdf(target)