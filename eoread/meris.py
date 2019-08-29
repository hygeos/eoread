#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
MERIS level1 reader

l1 = Level1_MERIS('MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1')
'''

import epr
import xarray as xr
import dask.array as da
import numpy as np
from threading import Lock
from os.path import join, basename, exists, dirname
import pandas as pd

from .naming import naming
from .common import len_slice, DataArray_from_array, AtIndex
from . import eo


BANDS_MERIS = [412, 443, 490, 510, 560,
               620, 665, 681, 709, 754,
               760, 779, 865, 885, 900]


def Level1_MERIS(filename, split=False, chunksize=(300, 400)):
    '''
    Read a MERIS Level1 product as an `xarray.Dataset`.

    Arguments:
    - split (bool): whether the wavelength dependent variables should be split in multiple 2D variables
    '''
    bname = basename(filename)

    ds = xr.Dataset()

    # epr api is not thread safe:
    # we have to use a lock for safe file access
    lock = Lock()

    prod = epr.Product(filename)
    ds.attrs[naming.totalwidth] = prod.get_scene_width()
    ds.attrs[naming.totalheight] = prod.get_scene_height()

    # read latitude, longitude, geometry, and TOA radiance
    for (name, param) in [
            (naming.lat, 'latitude'),
            (naming.lon, 'longitude'),
            (naming.sza, 'sun_zenith'),
            (naming.vza, 'view_zenith'),
            (naming.saa, 'sun_azimuth'),
            (naming.vaa, 'view_azimuth'),
            ('detector_index', 'detector_index'),
        ] + [(naming.Ltoa+f'_{b}', f'Radiance_{i+1}')
             for (i, b) in enumerate(BANDS_MERIS)]:
        ds[name] = DataArray_from_array(
            READ_MERIS(prod.get_band(param), lock),
            naming.dim2,
            chunksize
        )
    
    # Read attributes
    mph = prod.get_mph()
    for fname in mph.get_field_names():
        ds.attrs[fname] = mph.get_field(fname).get_elem()

    # TODO: parametrize dir_smile
    dir_smile = join(dirname(dirname(__file__)), 'auxdata', 'meris')
    assert exists(dir_smile), dir_smile
    if bname.startswith('MER_RR'):
        file_sun_spectral_flux = 'sun_spectral_flux_rr.txt'
    elif bname.startswith('MER_FR'):
        file_sun_spectral_flux = 'sun_spectral_flux_fr.txt'
    else:
        raise Exception(f'Error, could not identify whether MERIS file is RR or FR ({bname})')
    F0 = pd.read_csv(join(dir_smile, file_sun_spectral_flux), delimiter='\t').to_xarray()

    assert len(F0) == len(BANDS_MERIS) + 1

    for i, b in enumerate(BANDS_MERIS):
        ds[f'F0_{b}'] = DataArray_from_array(
            AtIndex(
                F0[f'E0_band{i}'],
                ds.detector_index,
                'index'),
            naming.dim2,
            chunksize,
        )
    # ds['F0'].attrs.update(ds.solar_flux.attrs)

    if not split:
        ds = eo.merge(
            ds,
            [a for a in ds if a.startswith(naming.Ltoa+'_')],
            naming.Ltoa,
            naming.bands,
        )
        ds = eo.merge(
            ds,
            [a for a in ds if a.startswith('F0_')],
            'F0',
            naming.bands,
        )
    
    return ds


class READ_MERIS:
    '''
    An array-like to read data from a given MERIS band
    '''
    def __init__(self, band, lock):
        self.width = band.product.get_scene_width()
        self.height = band.product.get_scene_height()
        self.band = band
        self.lock = lock
        self.shape = (self.height, self.width)
        self.dtype = {
            'float': np.float32,
            'short': np.int16,
        }[epr.data_type_id_to_str(band.data_type)]
        self.ndim = 2
    
    def __getitem__(self, keys):
        ystep = keys[0].step if keys[0].step is not None else 1
        xstep = keys[1].step if keys[1].step is not None else 1
        
        with self.lock:
            r = self.band.read_as_array(
                yoffset=keys[0].start,
                xoffset=keys[1].start,
                height=len_slice(keys[0], self.width),
                width=len_slice(keys[1], self.height),
                xstep=xstep,
                ystep=ystep,
            )
        assert r.dtype == self.dtype
        return r
                    	
