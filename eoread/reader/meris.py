#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
MERIS level1 reader

l1 = Level1_MERIS('MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1')
'''


from datetime import datetime
from pathlib import Path
from threading import Lock

import epr
import numpy as np
import pandas as pd
import xarray as xr

from core.tools import merge, raiseflag
from ..common import AtIndex, DataArray_from_array, len_slice, convert_for_nc
from ..utils.naming import naming, flags

BANDS_MERIS = [412, 443, 490, 510, 560,
               620, 665, 681, 709, 754,
               760, 779, 865, 885, 900]

central_wavelength_meris = {
        412: 412.691 , 443: 442.559,
        490: 489.882 , 510: 509.819,
        560: 559.694 , 620: 619.601,
        665: 664.573 , 681: 680.821,
        709: 708.329 , 754: 753.371,
        760: 761.508 , 779: 778.409,
        865: 864.876 , 885: 884.944,
        900: 900.000 ,
        }


def Level1_MERIS(filename,
                 dir_smile=None,
                 split=False,
                 chunks=500):
    '''
    Read a MERIS Level1 product as an `xarray.Dataset`.

    Arguments:
    ----------

    filename: str
        path to MERIS file
        (ex: 'MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1')
    dir_smile: str, default: '../auxdata/meris/'
        relative path to MERIS per-detector characterization
    split: bool
        whether the wavelength dependent variables should be split in multiple 2D variables
    chunks: int
        chunk size for dask array
    '''
    filename = Path(filename)
    bname = filename.name

    ds = xr.Dataset()

    # epr api is not thread safe:
    # we have to use a lock for safe file access
    lock = Lock()

    prod = epr.Product(str(filename))
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
            chunks=chunks,
        )

    if dir_smile is None:
        dir_smile = Path(__file__).parent/'auxdata'/'meris'
    else:
        dir_smile = Path(dir_smile)
    assert dir_smile.exists(), dir_smile
    if bname.startswith('MER_RR'):
        res = 'rr'
    elif bname.startswith('MER_FR'):
        res = 'fr'
    else:
        raise Exception(f'Error, could not identify whether MERIS file is RR or FR ({bname})')

    file_sun_spectral_flux = dir_smile/f'sun_spectral_flux_{res}.txt'
    file_detector_wavelength = dir_smile/f'central_wavelen_{res}.txt'
    F0 = pd.read_csv(file_sun_spectral_flux,
                     dtype='float32',
                     delimiter='\t').to_xarray()
    detector_wavelength = pd.read_csv(file_detector_wavelength,
                                      delimiter='\t').to_xarray()

    assert len(F0) == len(BANDS_MERIS) + 1
    assert len(detector_wavelength) == len(BANDS_MERIS) + 1

    for i, b in enumerate(BANDS_MERIS):
        ds[f'F0_{b}'] = DataArray_from_array(
            AtIndex(
                F0[f'E0_band{i}'],
                ds.detector_index,
                'index'),
            naming.dim2,
            chunks=chunks,
        )
        ds[f'wav_{b}'] = DataArray_from_array(
            AtIndex(
                detector_wavelength[f'lam_band{i}'],
                ds.detector_index,
                'index'),
            naming.dim2,
            chunks=chunks,
        )

    if not split:
        ds = merge(ds, dim=naming.bands)

    ds = ds.assign_coords(bands=BANDS_MERIS)

    # central (nominal) wavelength
    ds[naming.cwav] = xr.DataArray(
        np.array([central_wavelength_meris[b] for b in ds.bands.data],
                 dtype='float32'),
        dims=('bands',))

    #
    # Read attributes
    #
    mph = prod.get_mph()
    for fname in mph.get_field_names():
        ds.attrs[fname] = convert_for_nc(mph.get_field(fname).get_elem())

    ds.attrs[naming.platform] = 'ENVISAT'
    ds.attrs[naming.sensor] = 'MERIS'
    ds.attrs[naming.product_name] = ds.attrs['PRODUCT']
    ds.attrs[naming.input_directory] = str(filename.parent)

    # Read date
    dstart = read_date(mph, 'SENSING_START')
    dstop = read_date(mph, 'SENSING_STOP')
    d = dstart + (dstop - dstart)//2
    ds.attrs[naming.datetime] = d.isoformat()

    #
    # Ancillary data
    #
    aux = {}
    for name in ['zonal_wind',
                 'merid_wind',
                 'ozone',
                 'atm_press']:
        aux[name] = DataArray_from_array(
            READ_MERIS(prod.get_band(name), lock),
            naming.dim2,
            chunks=chunks,
        )
        aux[name].attrs['unit'] = prod.get_band(name).unit
        aux[name].attrs['description'] = prod.get_band(name).description

    ds[naming.horizontal_wind] = np.sqrt(
        aux['zonal_wind']**2 + aux['merid_wind']**2)
    ds[naming.total_column_ozone] = aux['ozone']
    ds[naming.sea_level_pressure] = aux['atm_press']

    #
    # Flags
    #
    ds[naming.flags] = xr.zeros_like(ds[naming.lat],
                                     dtype=naming.flags_dtype)
    for (flag, bmexpr) in [
            ('LAND', 'l1_flags.LAND_OCEAN'),
            ('L1_INVALID', '(l1_flags.INVALID) OR (l1_flags.SUSPECT) OR (l1_flags.COSMETIC)'),
        ]:
        raiseflag(
            ds[naming.flags],
            flag,
            flags[flag],
            DataArray_from_array(
                READ_BITMASK(prod, bmexpr, lock),
                naming.dim2,
                chunks=chunks,
            ),
        )
    
    return ds


def read_date(mph, field):
    dat = mph.get_field(field).get_elem(0)
    dat = dat.decode('utf-8')
    dat = dat.replace('-JAN-', '-01-')  # NOTE:
    dat = dat.replace('-FEB-', '-02-')  # parsing with '%d-%b-%Y...' may be
    dat = dat.replace('-MAR-', '-03-')  # locale-dependent
    dat = dat.replace('-APR-', '-04-')
    dat = dat.replace('-MAY-', '-05-')
    dat = dat.replace('-JUN-', '-06-')
    dat = dat.replace('-JUL-', '-07-')
    dat = dat.replace('-AUG-', '-08-')
    dat = dat.replace('-SEP-', '-09-')
    dat = dat.replace('-OCT-', '-10-')
    dat = dat.replace('-NOV-', '-11-')
    dat = dat.replace('-DEC-', '-12-')
    return datetime.strptime(dat, '%d-%m-%Y %H:%M:%S.%f')


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
        self.ndim = len(self.shape)

    def __getitem__(self, keys):
        assert len(keys) == self.ndim
        start = []
        steps = []
        sizes = []
        sel = []
        for k in keys:
            if isinstance(k, slice):
                st = k.start or 0
                start.append(st)
                steps.append(k.step or 1)
                sizes.append(k.stop - st)
                sel.append(slice(None))
            else:  # Indexing with int
                start.append(0)
                steps.append(1)
                sizes.append(1)
                sel.append(0)

        with self.lock:
            if (sizes[0] > steps[0]) and (sizes[1] > steps[1]):
                r = self.band.read_as_array(
                    yoffset=start[0],
                    xoffset=start[1],
                    height=sizes[0],
                    width=sizes[1],
                    ystep=steps[0],
                    xstep=steps[1],
                )
            else:
                r = self.band.read_as_array(
                    yoffset=start[0],
                    xoffset=start[1],
                    height=sizes[0],
                    width=sizes[1],
                )[::steps[0], ::steps[1]]
        assert r.dtype == self.dtype
        return r[sel[0], sel[1]]


class READ_BITMASK:
    '''
    An array-like to read MERIS bitmask
    '''
    def __init__(self, prod, bmexpr, lock):
        self.width = prod.get_scene_width()
        self.height = prod.get_scene_height()
        self.prod = prod
        self.lock = lock
        self.bmexpr = bmexpr
        self.shape = (self.height, self.width)
        self.ndim = len(self.shape)
        self.dtype = np.bool

    def __getitem__(self, keys):
        width = len_slice(keys[1], self.width)
        height = len_slice(keys[0], self.height)
        raster = epr.create_bitmask_raster(
            width, height,
            xstep=keys[1].step or 1,
            ystep=keys[0].step or 1,
            )
        with self.lock:
            self.prod.read_bitmask_raster(
                self.bmexpr,
                xoffset=keys[1].start or 0,
                yoffset=keys[0].start or 0,
                raster=raster)

        return raster.data

