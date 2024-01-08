import pandas as pd
import numpy as np

import shutil

from pathlib import Path
from xarray import Dataset

class Nomenclature:
    """
    Convert names according to a nomenclature csv file of the form:
    
    new_names, provider1, prodiver2, ..., providerX
    must be initialized with a provider name
    """
    
    def __init__(self, provider: str, csv_file = None, log: Path = None):
        
        if csv_file is None:
            self.csv_file = Path(__file__).parent / 'nomenclature.csv' # file path relative to the module
        else:
            self.csv_file = Path(csv_file)
        
        # verify that the csv_file exists
        assert self.csv_file.is_file()

        self.names = pd.read_csv(self.csv_file, skipinitialspace=True)
        self.names = self.names.apply(lambda x: x.str.strip() if x.dtype == 'object' else x) # remove trailing whitespaces
        
        # verify that provider is valid
        if provider not in list(self.names):
            raise ValueError(f'Cannot find column named {provider} in config file {self.file}')
            
        self.provider = provider
        self.log = log
        
        
    @staticmethod
    def copy_nomenclature_csv(target_dir='.'):
        """
        Create a local copy of the nomenclature CSV table, to the specified directory
        """
        
        src_file = Path(__file__).parent / 'nomenclature.csv' # file path relative to the module
        
        assert Path(target_dir).exists()
        
        shutil.copy(src_file, target_dir)
    
    
    def rename_dataset(self, ds) -> Dataset:
        """
        return the dataset with its variables renamed according to the csv file
        and the provider the calss was instanced with
        """
        
        new_names = {} # dictionnary of the form: {old_name: new_name, ... }
        for var in list(ds):
            new_names[var] = self.get_new_name(var)
        
        return ds.rename(new_names)
    
    
    def get_shortname(self, variable: str):
        
        if variable not in self.names['VARIABLE'].values:
            raise LookupError(f'Could not find any match for variable \'{variable}\' in column \'VARIABLE')
            
        if len(self.names[self.names['VARIABLE'] == variable].values) > 1:
            raise LookupError(f'Amigous definition for var \'{variable}\' in column \'VARIABLE\'')
            
        short_name = self.names[self.names['VARIABLE'] == variable][self.provider].values
        
        if len(short_name) == 0:
            raise LookupError(f'Could not find any match for variable \'{variable}\' in column \'CAMS\'')
        
        if len(short_name) > 1:
            raise LookupError(f'Several definitions for var \'{variable}\' in column \'{self.provider}\'')
        
        short_name = short_name[0]
        
        if not type(short_name) == str:
            raise LookupError(f'Could not find any match for variable \'{variable}\' in column \'{self.provider}\'')
        
        return short_name
    
    
    def assert_shortname_is_defined(self, var: str) -> bool:
        """
        Check wether of not a variable was defined in the csv file, for the provider the class
        has been instanced with.
        Allow to catch errors before downloading / renaming dataset
        """
        
        if var not in list(self.names[self.provider]): 
            raise KeyError(f'Variable {var} equivalents have not been specified in the config file')
        
        return 
        
        
    def get_new_name(self, var: str) -> str:
        """
        Return the corresponding new name for a variable depending on the provider
        the class was instanced with
        """
            
        # if not in column raise error
        if var not in list(self.names[self.provider]): 
            raise KeyError(f'Variable {var} equivalents have not been specified in the config file')
            
        # get equivalent for variable name
        return self.names[self.names[self.provider] == var]['VARIABLE'].values[0].strip()
        