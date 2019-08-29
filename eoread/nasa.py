#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read NASA Level1 files from MODIS, VIIRS, SeaWiFS

Use the L1C approach: L1C files are generated with SeaDAS (l2gen) to
get all radiometric correction
"""

import xarray as xr


def Level1_NASA(filename, chunks=500):
    ds = xr.open_dataset(filename, chunks=chunks)

    print(ds)

    return ds

