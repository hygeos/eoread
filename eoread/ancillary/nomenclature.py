import pandas as pd

from pathlib import Path
from xarray import Dataset

class Nomenclature:
    """
    Convert names according to a nomenclature csv file of the form:
    
    new_names, provider1, prodiver2, ..., provider7 [, Unit]
    must be initialized with a provider name
    """
    
    def __init__(self, provider: str, log: Path = None):
        
        self.csv_file = Path(__file__).parent / 'nomenclature.csv' # file path relative to the module

        self.names = pd.read_csv(self.csv_file, skipinitialspace=True)
        self.names = self.names.apply(lambda x: x.str.strip() if x.dtype == 'object' else x) # remove trailing whitespaces
        
        # verify that provider is valid
        if provider not in list(self.names):
            raise ValueError(f'Cannot find column named {provider} in config file {self.file}')
            
        self.provider = provider
        self.log = log
    
    
    def rename_dataset(self, ds) -> Dataset:
        """
        return the dataset with its variables renamed according to the csv file
        and the provider the calss was instanced with
        """
        
        new_names = {} # dictionnary of the form: {old_name: new_name, ... }
        for var in list(ds):
            new_names[var] = self.get_new_name(var)
        
        error = self.verify_units(ds)
        # invalid unit, conversion needed
        if error: raise ValueError('Invalid units encountered after post-processing')
        
        return ds.rename(new_names)
    
    
    def assert_var_is_defined(self, var: str) -> bool:
        """
        Check wether of not a variable was defined in the csv file, for the provider the class
        has been instanced with.
        Allow to catch errors before downloading / renaming dataset
        """
        
        if var not in list(self.names[self.provider]): 
            raise KeyError(f'Variable {var} equivalents have not been specified in the config file')
        
        return 
        
    def verify_units(self, ds):
        logs = ''
        err_logs = ''
        encountered_error = False
        
        for var in list(ds):
            expected = str(self.names[self.names[self.provider] == var]['UNITS'].values[0]).strip()
            actual = ''
            
            logs += '\n'
            
            # get actual unit
            if hasattr(ds[var], 'units'): 
                actual = ds[var].units
                
            # expected dimensionless
            if expected == 'nan':
                u = actual.lower().strip()
                
                if not hasattr(ds[var], 'units'): # valid # TODO Error when no units attribute ??
                    logs += f'[valid] {var}:\t units: dimensionless (no units attrs)'
                
                # dimensionless value variations
                elif (   u == '~' # CAMS            
                      or u == 'dimensionless'
                      or u == '(0 - 1)' # ERA5
                      or u == '1'       # MERRA2
                ):
                    logs += f'[valid] {var}:\t units: dimensionless'
                
                else: # invalid 
                    logs     += f'[ERROR] {var}:\t units: expected dimensionless got: {actual}, maybe you forgot to specify the expected unit'
                    err_logs += f'[ERROR] {var}:\t units: expected dimensionless got: {actual}, maybe you forgot to specify the expected unit'
                    encountered_error = True
                
            else: # dimensions are expected
                u = actual.strip().replace('**', '') # (era5) kg m**-2 ==  (merra2) kg m-2
                
                if actual == expected:
                    logs += f'[valid] {var}:\t units: {expected}'
                else:
                    logs     += f'[ERROR] {var}:\t units: expected {expected} got: {actual}'
                    err_logs += f'[ERROR] {var}:\t units: expected {expected} got: {actual}'
                    encountered_error = True
        
        if self.log: # only log if path provided
            with open(self.log, 'a+') as myfile:
                myfile.write(f'\n\n-------[{self.provider}]-------')    
                myfile.write(logs)
        
        # also use standard output to print errors
        if encountered_error: print(err_logs)
        
        return encountered_error
    
    
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
        