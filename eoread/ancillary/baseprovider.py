from typing import Callable
from pathlib import Path
from math import floor, ceil

from datetime import datetime, timedelta, date
import xarray as xr

from ..utils.static import interface, abstract
from .. import eo
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
        
        # dictionnary of computable variables
        self.computables: dict[str:tuple] = {}
        # ex: {'tot_opt_depth': (compute_tot_opt_depth, ['aod469', 'aod670'])}
        
    
    @interface
    def get_day(self, variables: list[str], date: date, area: list|None=None) -> xr.Dataset:
        """
        Download and apply post-process to the downloaded data for the given date
        Standardize the dataset according to the nomenclature module
        Return data of the full day
     
        - variables: list of variables names to download
        - d: date of desired the data
        - area: [90, -180, -90, 180] → [north, west, south, east]
        """
        # check for computed vars if not self.no_std
        vars_to_compute, vars_to_query = (None, variables) if self.no_std else self._find_computable_variables(variables)
        
        filepath = self.download(vars_to_query, date, area)
        ds = xr.open_mfdataset(filepath)                      # open dataset
        
        if self.no_std: # do not standardize, return as is
            return ds
        
        # standardizatoin
        ds = self._compute_variables(vars_to_compute, ds)
        
        return self.standardize(ds) # standardize according to nomenclature file
    
    
    @interface
    def get(self, variables: list[str], dt: datetime, area: list|None=None) -> xr.Dataset:
        """
        Download and apply post-process to the downloaded data for the given date
        Standardize the dataset according to the nomenclature module
        Return data interpolated on time=dt
     
        - variables: list of variables names to download
        - d: datetime of desired the data
        - area: [90, -180, -90, 180] → [north, west, south, east]
        """
        
        # check for computed vars if not self.no_std
        vars_to_compute, vars_to_query = (None, variables) if self.no_std else self._find_computable_variables(variables)
        
        # get data
        day = date(dt.year, dt.month, dt.day) # get the day wether dt is a datetime or a datetime
        filepath = self.download(vars_to_query, day, area)
        ds = xr.open_mfdataset(filepath)
        
        # download next day if necessary
        if dt.hour == 23:
            next_day = day + timedelta(days=1)
            filepath = self.download(vars_to_query, next_day, area) # download next day
            ds2 = xr.open_mfdataset(filepath) # open dataset
            ds = xr.concat([ds, ds2], dim='time') # concatenate
        
        ds = ds.interp(time=dt) # interpolate on time
        
        if self.no_std: # do not standardize, return as is
            return ds
        
        # standardizatoin
        ds = self._compute_variables(vars_to_compute, ds)
        
        return self.standardize(ds) # standardize according to nomenclature file
    
    
    @interface
    def get_range(self, variables: list[str], date_start: date, date_end: date, area: list|None=None) -> xr.Dataset:
        """
        Download and apply post-process to the downloaded data for the dates between date_start and date_end
        Standardize the dataset according to the nomenclature module
     
        - variables: list of variables names to download
        - d: datetime of desired the data
        - area: [90, -180, -90, 180] → [north, west, south, east]
        """
        
        days_delta: int = (date_end - date_start).days
        
        assert days_delta >= 0, 'date_start must be anterior to date_end'
        
        ds = self.get_day(variables, date=date_start, area=area)
        
        for i in range(1, days_delta+1):
            cday = date_start + timedelta(days=i)
        
            ds_day = self.get_day(variables, date=cday, area=area)
            
            ds = xr.concat([ds, ds_day], dim='time')
        
        return ds
    
    
    def _find_computable_variables(self, variables) -> tuple[list, list]:
        """
        Analyse the list of variables to separate computed variables from queried variables 
        (with added dependencies for the computation)
        returns (computed, queried)

        names are still standardized to not interfere with previous get() functionning
        """
        shortname: Callable = self.names.get_shortname
        
        computed = [v for v in variables if shortname(v) in self.computables]
        queried  = [v for v in variables if v not in computed]
        
        dependencies = []
        for v in computed:
            func, inputs = self.computables[shortname(v)] 
            dependencies += inputs
        
        dependencies = [self.names.get_new_name(v) for v in dependencies]    
        
        for v in dependencies:
            if shortname(v) in self.computables:
                raise RecursionError('Cannot (yet) have computed variables as dependencies for other computed variables')
            
        # add computation dependencies to query        
        queried = list(set(queried + dependencies)) # remove duplicates
        
        return computed, queried
    
    
    def _compute_variables(self, variables, ds):
        
        if len(variables) == 0: # return early if no var to compute
            return ds
        
        var = [(v, self.names.get_shortname(v)) for v in variables]
        
        for tup in var:
            std_name, short_name = tup
            
            if short_name not in self.computables:
                raise KeyError(f'Could not find compute instructions for variable: {short_name}')
            
            # get function and call it with ds (which contains the params, which are dependencies)
            func, params = self.computables[short_name]
            ds = func(ds, short_name)
            
            ds[short_name].attrs['origin'] = f'Computed from {params} provided by {self.name}'
            ds[short_name].attrs['history'] = f'Computed from {params} by function \'{func.__name__}\' called by provider class \'{self.name}\' from package \'eoread.ancillary\''
            
        return ds
    
    
    @abstract # to be defined by subclasses
    def standardize(self, ds: xr.Dataset) -> xr.Dataset:
        pass
        
    
    def _get_filename(self, variables: list[str], d: date, acronym: str, area: list|None) -> str:
        """
        Constructs and return the target filename according to the nomenclature specified
        in the attribute 'filename_pattern'
        """
        
        area_str = "global"
        if area is not None and area != [90, -180, -90, 180]:
            
            area = [
                ceil(area[0]),
                floor(area[1]),
                floor(area[2]),
                ceil(area[3])
            ]
            area_str = f"region_{area[0]}_{area[1]}_{area[2]}_{area[3]}"
        
        # construct chain of variables short name 
        vars = variables.copy()
        vars.sort()
        
        vars_str = vars.pop(0)  # get first element without delimiter
        for v in vars: 
            vars_str += '_' + v # apply delimiter and names
            
        d = d.strftime('%Y%m%d')
        return self.file_pattern % (acronym, area_str, vars_str, d)
    