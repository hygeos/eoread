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
    Function that detect and read level 1 product file with the appropriated reader
    """
    # Detect product type
    dict_pattern = get_pattern(path.stem)
    assert get_level(path.stem, dict_pattern) == 1, \
        'Path does not correspond to a level 1 product'
    
    # Import reader and read provided file
    assert dict_pattern['reader'] != '', f"No reader exists for {dict_pattern['Name']}"
    to_import = dict_pattern['reader'].split()
    module = importlib.import_module(to_import[0])
    try: reader = getattr(module, to_import[1].replace('Level','Level1'))
    except AttributeError as e: log.error(f'Reader importaton failed, got {e}') 
    
    return reader(path, **kwargs)


def Level2(path: Path, **kwargs) -> xr.Dataset:
    """
    Function that detect and read level 2 product file with the appropriated reader
    """
    # Detect product type
    dict_pattern = get_pattern(path.stem)
    assert get_level(path.stem, dict_pattern) == 2, \
        'Path does not correspond to a level 2 product'
    
    # Import reader and read provided file
    assert dict_pattern['reader'] != '', f"No reader exists for {dict_pattern['Name']}"
    to_import = dict_pattern['reader'].split()
    module = importlib.import_module(to_import[0])
    try: reader = getattr(module, to_import[1].replace('Level','Level2'))
    except AttributeError as e: log.error(f'Reader importaton failed, got {e}') 
    
    return reader(path, **kwargs)