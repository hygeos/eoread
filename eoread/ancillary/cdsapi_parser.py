# -*- coding: utf-8 -*-

from pathlib import Path
import os

def read_multi_config(path: Path):
    """
    Read file from path values and parses configuration for CDSAPI
    
    name.var: value
    ex: "cds.url: https://cds.climate.copernicus.eu/api/v2"
        "cds.key: cds.key: 12434:34536a45-45d7-87c6-32a5-as9as8d7f6a9" # fake credentials
    """
    
    config = {}
    
    if isinstance(path, str): path = Path(path)
    if not path.exists(): raise RecursionError(f'Could not find file {path}') 
    
    path = path.resolve()
    f = open(path, 'r')
    
    for line in f.readlines():
        
        if ':' not in line: continue
        left, value = [s.strip() for s in line.strip().split(":", 1)]
        # ex: 'cds.url', ' https://cds.climate.copernicus.eu/api/v2'
        
        if '.' not in left: continue # skip line if no dot in left side
        name, var = [s.strip() for s in left.split('.', 1)]
        # ex: 'cds', 'url'
        
        if name not in config: config[name] = {}     
        
        if var in ['url', 'key']:
            config[name][var] = value
        else:   raise SyntaxWarning(f'Unrecognized key in file \'{path}\', ' 
                                           'only \'key\' and \'url\' will be used')
    
    f.close()
    
    if len(config) == 0: raise SyntaxError(f'Invalid configuration syntax in file \'{path}\'')
    
    return config

def read_config(namespace: str, path: Path):
    """
    Returns a specific namespace from read_multiple_config
    """
    cfg = read_multi_config(path)
    if namespace not in cfg: raise KeyError(f'Cannot find namespace \'{namespace}\' in config file \'{path.resolve()}\'')
    return cfg[namespace]