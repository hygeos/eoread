#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read NASA Level1 files from MODIS, VIIRS, SeaWiFS

Use the L1C approach: L1C files are generated with SeaDAS (l2gen) to
get all radiometric correction
"""

import xarray as xr
import numpy as np
from datetime import datetime
from os.path import dirname
import subprocess
from .download import download_url
from .naming import naming, flags
from .common import DataArray_from_array
from . import eo


def check_nasa_download(filename):
    '''
    sanity check of file downloaded on NASA earthdata
    check that downloaded file is not HTML
    raise an error if it is not the case (authentication error)
    '''
    errormsg = 'Error authenticating to NASA EarthData for downloading ancillary data. ' \
    'Please provide authentication through .netrc. See more information on ' \
    'https://support.earthdata.nasa.gov/index.php?/Knowledgebase/Article/View/43/21/how-to-access-urs-gated-data-with-curl-and-wget'
    with open(filename, 'rb') as fp:
        assert not fp.read(100).startswith(b'<!DOCTYPE html>'), errormsg


def nasa_download(product, dirname, tmpdir=None, verbose=True):
    '''
    Download a product on oceandata.sci.gsfc.nasa.gov

    Example:
        nasa_download('A2005005002500.L1A_LAC.bz2', '/data/')
    '''
    url = f'https://oceandata.sci.gsfc.nasa.gov/ob/getfile/{product}'
    return download_url(
        url,
        dirname,
        verbose=verbose,
        tmpdir=tmpdir,
        wget_opts='-nv --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --auth-no-challenge',
        check_function=check_nasa_download,
        lock_timeout=3600,
        )


def nasa_search(**kwargs):
    """
    Search for files on oceancolor server

    Args are passed directly to the query

    Example:

    nasa_search(sensor='seawifs',
                sdate='2000-04-17',
                edate='2000-04-17',
                dtype='L1',
                search='*L1A_GAC'):

    See https://oceancolor.gsfc.nasa.gov/data/download_methods/#api
    """
    query = [f'{k}={v}' for k, v in kwargs.items()]
    query += ['addurl=0', 'results_as_file=1']

    query_str = '&'.join(query)
    cmd = f'wget -q --post-data="{query_str}" -O - https://oceandata.sci.gsfc.nasa.gov/api/file_search'
    return subprocess.check_output(cmd, shell=True).decode().split()


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

