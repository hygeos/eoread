from typing import Callable
from pathlib import Path
from math import floor, ceil

from datetime import datetime, timedelta, date
import xarray as xr

from eoread.static import interface, abstract
from eoread import eo
from .nomenclature import Nomenclature


@abstract
class BaseProvider:
    
    @interface
    def __init__(self, name: str, model: Callable, directory: Path, nomenclature_file=None, 
                 offline: bool=False, verbose: bool=True, no_std: bool=False):
        self.name = name
        
        self.directory = directory.resolve()
        if not self.directory.exists():
            raise FileNotFoundError(f'Directory "{self.directory}" does not exist. Use an existing directory.')
        
        if not callable(model):
            raise ValueError(f"Invalid model parameter, please use any of the functions stored in {self.name}.models")
            
        self.model = model
        
        self.offline = offline
        self.verbose = verbose
        self.no_std = no_std
        
        self.file_pattern = f"{self.name}_%s_%s_%s_%s.nc" # model_acronym, 'global'/'region', vars, date
        
        # General variable nomenclature preparation
        self.names = Nomenclature(provider=name, csv_file=nomenclature_file)
    
    @interface
    def get_day(self, variables: list[str], date: date, area: list=[90, -180, -90, 180]) -> xr.Dataset:
        """
        Download and apply post-process to the downloaded data for the given date
        Standardize the dataset according to the nomenclature module
        Return data of the full day
     
        - variables: list of strings of the model variables short names to download ex: ['gtco3', 'aod550', 'parcs'] TODO change
        - d: date of desired the data
        - area: [90, -180, -90, 180] → [north, west, south, east]
        """
        
        # day = date(day.year, day.month, day.day) # get the day wether dt is a datetime or a date
        
        filepath = self.download(variables, date, area)
        ds = xr.open_mfdataset(filepath)                      # open dataset
        
        if self.no_std: # do not standardize, return as is
            return ds
        return self.standardize(ds) # standardize according to nomenclature file
    
    
    @interface
    def get(self, variables: list[str], dt: datetime, area: list=[90, -180, -90, 180]) -> xr.Dataset:
        """
        Download and apply post-process to the downloaded data for the given date
        Standardize the dataset according to the nomenclature module
        Return data interpolated on time=dt
     
        - variables: list of strings of the model variables short names to download ex: ['gtco3', 'aod550', 'parcs'] TODO change
        - d: datetime of desired the data
        - area: [90, -180, -90, 180] → [north, west, south, east]
        """
        
        # get data
        day = date(dt.year, dt.month, dt.day) # get the day wether dt is a datetime or a datetime
        filepath = self.download(variables, day, area)
        ds = xr.open_mfdataset(filepath)   
        
        # download next day if necessary
        if dt.hour == 23:
            next_day = day + timedelta(days=1)
            filepath = self.download(variables, next_day.date(), area) # download next day
            ds2 = xr.open_mfdataset(filepath)                           # open dataset
            ds = xr.concat([ds, ds2], dim='time') # concatenate
        
        ds = ds.interp(time=dt) # interpolate on time
        
        if self.no_std: # do not standardize, return as is
            return ds
        return self.standardize(ds) # standardize according to nomenclature file
    
    
    @abstract # to be defined by subclasses
    def standardize(self, ds: xr.Dataset) -> xr.Dataset:
        pass
        
    
    def _get_filename(self, variables: list[str], d: date, acronym: str, area) -> str:
        """
        Constructs and return the target filename according to the nomenclature specified
        in the attribute 'filename_pattern'
        """
        
        area_str = "global"
        if area != [90, -180, -90, 180]:
            
            area = [
                ceil(area[0]),
                floor(area[1]),
                floor(area[2]),
                ceil(area[3])
            ]
            area_str = f"region_{area[0]}-{area[1]}-{area[2]}-{area[3]}"
        
        # construct chain of variables short name 
        vars = variables.copy()
        vars.sort()
        
        vars_str = vars.pop(0)  # get first element without delimiter
        for v in vars: 
            vars_str += '_' + v # apply delimiter and names
            
        d = d.strftime('%Y%m%d')
        return self.file_pattern % (acronym, area_str, vars_str, d)
    