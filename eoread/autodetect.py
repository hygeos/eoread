#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic product loader with autodetection
"""

from core.geo import get_pattern, get_level
from pathlib import Path
from core import log

import xarray as xr
import importlib


def Level1(path: Path, **kwargs) -> xr.Dataset:
    """
    Automatically detect and read a Level 1 satellite product using the appropriate reader.
    
    This function analyzes the filename pattern to determine the sensor type and
    dispatches to the corresponding Level1 reader function.
    
    Parameters
    ----------
    path : Path
        Path to the satellite product file or directory.
    **kwargs : dict
        Additional arguments passed to the specific reader function.
        
    Returns
    -------
    xr.Dataset
        Loaded Level 1 product data.
        
    Raises
    ------
    AssertionError
        If the path does not correspond to a Level 1 product or
        if no reader exists for the detected product type.
        
    Examples
    --------
    >>> from pathlib import Path
    >>> ds = Level1(Path('S2A_MSIL1C_*.SAFE'))
    >>> ds = Level1(Path('MER_FRS_*.N1'), chunks=1000)
    """
    # Detect product type
    dict_pattern = get_pattern(path.stem)
    assert get_level(path.stem, dict_pattern) == 1, \
        'Path does not correspond to a level 1 product'
    
    # Import reader and read provided file
    assert dict_pattern['reader'] != '', f'No reader exists for {dict_pattern['Name']}'
    to_import = dict_pattern['reader'].split()
    module = importlib.import_module(to_import[0])
    try: reader = getattr(module, to_import[1].replace('Level','Level1'))
    except AttributeError as e: log.error(f'Reader importaton failed, got {e}') 
    
    return reader(path, **kwargs)


def Level2(path: Path, **kwargs) -> xr.Dataset:
    """
    Automatically detect and read a Level 2 satellite product using the appropriate reader.
    
    This function analyzes the filename pattern to determine the sensor type and
    dispatches to the corresponding Level2 reader function.
    
    Parameters
    ----------
    path : Path
        Path to the satellite product file or directory.
    **kwargs : dict
        Additional arguments passed to the specific reader function.
        
    Returns
    -------
    xr.Dataset
        Loaded Level 2 product data.
        
    Raises
    ------
    AssertionError
        If the path does not correspond to a Level 2 product or
        if no reader exists for the detected product type.
        
    Examples
    --------
    >>> from pathlib import Path
    >>> ds = Level2(Path('S3A_OL_2_WFR____*.SEN3'))
    """
    # Detect product type
    dict_pattern = get_pattern(path.stem)
    assert get_level(path.stem, dict_pattern) == 2, \
        'Path does not correspond to a level 2 product'
    
    # Import reader and read provided file
    assert dict_pattern['reader'] != '', f'No reader exists for {dict_pattern['Name']}'
    to_import = dict_pattern['reader'].split()
    module = importlib.import_module(to_import[0])
    try: reader = getattr(module, to_import[1].replace('Level','Level2'))
    except AttributeError as e: log.error(f'Reader importaton failed, got {e}') 
    
    return reader(path, **kwargs)