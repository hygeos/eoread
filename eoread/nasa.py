#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read NASA Level1 files from MODIS, VIIRS, SeaWiFS

Use the L1C approach: L1C files are generated with SeaDAS (l2gen) to
get all radiometric correction
"""

import xarray as xr
from .naming import naming, flags
from .common import DataArray_from_array
from . import eo
import numpy as np
from datetime import datetime
from os.path import dirname


def Level1_NASA(filename, chunks=500):
    ds = xr.open_dataset(filename, chunks=chunks)

    dstart = datetime.strptime(ds.attrs['time_coverage_start'], "%Y-%m-%dT%H:%M:%S.%fZ")
    dstop = datetime.strptime(ds.attrs['time_coverage_end'], "%Y-%m-%dT%H:%M:%S.%fZ")
    d = dstart + (dstop - dstart)//2
    ds.attrs[naming.datetime] = d.isoformat()
    ds.attrs[naming.sensor] = ds.attrs['instrument']
    ds.attrs[naming.input_directory] = dirname(filename)

    sensor_band = xr.open_dataset(filename, group='/sensor_band_parameters', chunks=chunks)
    bands = sensor_band['wavelength'].values[sensor_band.number_of_reflective_bands.values].astype('int32')
    ds[naming.wav] = np.array(bands, dtype='float32')

    navi = xr.open_dataset(filename, group='navigation_data', chunks=chunks)
    navi = navi.rename_dims({'number_of_lines':naming.rows, 'pixel_control_points':naming.columns})
    ds[naming.lat] = DataArray_from_array(navi.latitude.values.astype('float32'), naming.dim2, chunks=chunks)
    ds[naming.lon] = DataArray_from_array(navi.longitude.values.astype('float32'), naming.dim2, chunks=chunks)
    
    geo_data = xr.open_dataset(filename, group='/geophysical_data', chunks=chunks)
    geo_data = geo_data.rename_dims({'number_of_lines':naming.rows, 'pixels_per_line':naming.columns})
    for n,r,p in [(naming.Rtoa+f'_{b}', f'rhot_{b}', f'polcor_{b}') for b in bands]:
        try:
            ds[n] = geo_data[r]/geo_data[p]
        except:
            pass

    for (name, param) in [(naming.sza, 'solz'),
            (naming.vza, 'senz'),
            (naming.saa, 'sola'),
            (naming.vaa, 'sena'),
            ]:
        ds[name] = geo_data[param]

    eo.init_geometry(ds)

    ds[naming.flags] = xr.zeros_like(ds[naming.lat], dtype=naming.flags_dtype)
    for (flag, flag_list) in [('LAND',['LAND']), ('L1_INVALID',['ATMFAIL','PRODFAIL'])]:
        flag_value = 0
        for f in flag_list:
            flag_value += geo_data.l2_flags.flag_masks[geo_data.l2_flags.flag_meanings.split().index(f)]

        eo.raiseflag(ds[naming.flags],flag, flags[flag], DataArray_from_array((geo_data.l2_flags&flag_value!=0), naming.dim2, chunks=chunks))

    ds = eo.merge(ds, dim=naming.bands)
    return ds

