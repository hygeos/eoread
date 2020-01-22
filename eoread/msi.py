#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
List of MSI bands:
-----------------

Band Use         Wavelength Resolution
B1   Aerosols    443nm      60m
B2   Blue        490nm      10m
B3   Green       560nm      10m
B4   Red         665nm      10m
B5   Red Edge 1  705nm      20m
B6   Red Edge 2  740nm      20m
B7   Red Edge 3  783nm      20m
B8   NIR         842nm      10m
B8a  Red Edge 4  865nm      20m
B9   Water vapor 940nm      60m
B10  Cirrus      1375nm     60m
B11  SWIR 1      1610nm     20m
B12  SWIR 2      2190nm     20m
'''


import os
from datetime import datetime
from glob import glob

import dask.array as da
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from lxml import objectify

from . import eo
from .common import DataArray_from_array, Interpolator, Repeat
from .naming import naming, flags

msi_band_names = {
        443 : 'B01', 490 : 'B02',
        560 : 'B03', 665 : 'B04',
        705 : 'B05', 740 : 'B06',
        783 : 'B07', 842 : 'B08',
        865 : 'B8A', 945 : 'B09',
        1375: 'B10', 1610: 'B11',
        2190: 'B12',
        }


def Level1_MSI(dirname,
               resolution='60',
               geometry=True,
               chunks=500,
               split=False):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances,
    the angles on the full grid, etc.

    Arguments:
        resolution: '60', '20' or '10' (in m)
        geometry: whether to read the geometry
        split: whether the wavelength dependent variables should be split in multiple 2D variables
    '''
    ds = xr.Dataset()
    dirname = os.path.abspath(dirname)

    if dirname.endswith('.SAFE'):
        granules = glob(os.path.join(dirname, 'GRANULE', '*'))
        assert len(granules) == 1
        granule_dir = granules[0]
    else:
        granule_dir = dirname

    # load xml file
    xmlfiles = glob(os.path.join(granule_dir, '*.xml'))
    assert len(xmlfiles) == 1
    xmlfile = xmlfiles[0]
    xmlgranule = objectify.parse(xmlfile).getroot()

    # load main xml file
    xmlfile = os.path.join(os.path.dirname(os.path.dirname(granule_dir)), 'MTD_MSIL1C.xml')
    xmlroot = objectify.parse(xmlfile).getroot()
    quantif = float(xmlroot.General_Info.find('Product_Image_Characteristics').QUANTIFICATION_VALUE)

    # read date
    ds.attrs['datetime'] = str(xmlgranule.General_Info.find('SENSING_TIME'))
    geocoding = xmlgranule.Geometric_Info.find('Tile_Geocoding')
    tileangles = xmlgranule.Geometric_Info.find('Tile_Angles')

    # get platform
    tile_id = str(xmlgranule.General_Info.find('TILE_ID')[0])
    platform = tile_id[:3]
    assert platform in ['S2A', 'S2B']

    # read image size for current resolution
    for e in geocoding.findall('Size'):
        if e.attrib['resolution'] == str(resolution):
            ds.attrs[naming.totalheight] = int(e.find('NROWS').text)
            ds.attrs[naming.totalwidth] = int(e.find('NCOLS').text)
            break

    # attributes
    ds.attrs[naming.platform] = platform
    ds.attrs['resolution'] = resolution
    ds.attrs[naming.sensor] = 'MSI'
    ds.attrs[naming.product_name] = os.path.basename(dirname)
    ds.attrs[naming.input_directory] = os.path.dirname(dirname)

    # lat-lon
    msi_read_latlon(ds, geocoding, chunks)

    # msi_read_geometry
    if geometry:
        msi_read_geometry(ds, tileangles, chunks)

    # msi_read_toa
    ds = msi_read_toa(ds, granule_dir, quantif, split, chunks)

    # read spectral information
    msi_read_spectral(ds)

    # flags
    ds[naming.flags] = xr.zeros_like(
        ds.vza,
        dtype=naming.flags_dtype)
    eo.raiseflag(
        ds[naming.flags],
        'L1_INVALID',
        flags['L1_INVALID'],
        np.isnan(ds.vza)
        )

    return ds


def msi_read_latlon(ds, geocoding, chunks):
    ds[naming.lat] = DataArray_from_array(
        LATLON(geocoding, 'lat', ds),
        naming.dim2,
        chunks=chunks,
    )

    ds[naming.lon] = DataArray_from_array(
        LATLON(geocoding, 'lon', ds),
        naming.dim2,
        chunks=chunks,
    )


def msi_read_toa(ds, granule_dir, quantif, split, chunks):

    for k, v in msi_band_names.items():
        filenames = glob(os.path.join(granule_dir, 'IMG_DATA', f'*_{v}.jp2'))
        assert len(filenames) == 1
        filename = filenames[0]

        arr = xr.open_rasterio(
            filename,
            chunks=chunks,
        ).astype('float32')/quantif
        arr = arr.squeeze('band')
        arr = arr.drop('x').drop('y')

        xrat = len(arr.x)/float(ds.totalwidth)
        yrat = len(arr.y)/float(ds.totalheight)

        if xrat >= 1.:
            # downsample
            arr_resampled = 0.
            for i in range(int(xrat)):
                for j in range(int(yrat)):
                    arr_resampled += arr.isel(x=slice(i, None, int(xrat)),
                                              y=slice(j, None, int(yrat)))
            arr_resampled /= int(xrat)*int(yrat)
            arr_resampled = arr_resampled.drop('band')
        else:
            # over-sample
            arr_resampled = DataArray_from_array(
                Repeat(arr, (int(1/yrat), int(1/xrat))),
                ('y', 'x'),
                chunks,
            )

        arr_resampled = arr_resampled.rename({
            'x': naming.columns,
            'y': naming.rows})

        arr_resampled.attrs['bands'] = k
        arr_resampled.attrs['band_name'] = v
        ds[naming.Rtoa+f'_{k}'] = arr_resampled

    if not split:
        ds = eo.merge(ds, [a for a in ds if a.startswith(naming.Rtoa+'_')],
                      naming.Rtoa, naming.bands, coords=list(msi_band_names.keys()))

    return ds


def msi_read_spectral(ds):
    # read srf
    dir_aux_msi = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'auxdata', 'msi')
    platform = ds.attrs['platform']
    srf_file = os.path.join(
        dir_aux_msi,
        f'S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_{platform}.csv')

    assert os.path.exists(srf_file)

    srf_data = pd.read_csv(srf_file)
    wav = srf_data.SR_WL

    wav_data = []

    for b, bn in msi_band_names.items():
        col = platform + '_SR_AV_' + bn.replace('B0', 'B')
        srf = srf_data[col]
        wav_eq = np.trapz(wav*srf)/np.trapz(srf)
        wav_data.append(wav_eq)

    ds['wav'] = xr.DataArray(
        da.from_array(wav_data),
        dims=(naming.bands),
    )


def msi_read_geometry(ds, tileangles, chunks):

    # read solar angles at tiepoints
    sza = read_xml_block(tileangles.find('Sun_Angles_Grid').find('Zenith').find('Values_List'))
    saa = read_xml_block(tileangles.find('Sun_Angles_Grid').find('Azimuth').find('Values_List'))

    shp = (ds.totalheight, ds.totalwidth)

    # read view angles (for each band)
    vza = {}
    vaa = {}
    for e in tileangles.findall('Viewing_Incidence_Angles_Grids'):

        # read zenith angles
        data = read_xml_block(e.find('Zenith').find('Values_List'))
        bandid = int(e.attrib['bandId'])
        if bandid not in vza:
            vza[bandid] = data
        else:
            ok = ~np.isnan(data)
            vza[bandid][ok] = data[ok]

        # read azimuth angles
        data = read_xml_block(e.find('Azimuth').find('Values_List'))
        bandid = int(e.attrib['bandId'])
        if bandid not in vaa:
            vaa[bandid] = data
        else:
            ok = ~np.isnan(data)
            vaa[bandid][ok] = data[ok]

    # use the first band as vza and vaa
    k = sorted(vza.keys())[0]
    assert k in vaa

    # initialize the dask arrays
    for name, tie in [(naming.sza, sza),
                      (naming.saa, saa),
                      (naming.vza, vza[k]),
                      (naming.vaa, vaa[k]),
                      ]:
        da_tie = xr.DataArray(
            tie,
            dims=('tie_rows', 'tie_columns'),
            coords={'tie_rows': np.linspace(0, shp[0]-1, sza.shape[0]),
                    'tie_columns': np.linspace(0, shp[1]-1, sza.shape[1])})
        ds[name+'_tie'] = da_tie
        ds[name] = DataArray_from_array(
            Interpolator(shp, ds[name+'_tie']),
            naming.dim2,
            chunks,
        )


def read_xml_block(item):
    '''
    read a block of xml data and returns it as a numpy float32 array
    '''
    d = []
    for i in item.iterchildren():
        d.append(i.text.split())
    return np.array(d, dtype='float32')


class LATLON:
    '''
    An array-like to calculate the MSI lat-lon
    '''
    def __init__(self, geocoding, kind, ds):
        self.kind = kind

        code = geocoding.find('HORIZONTAL_CS_CODE').text

        self.proj = pyproj.Proj('+init={}'.format(code))

        # lookup position in the UTM grid
        for e in geocoding.findall('Geoposition'):
            if e.attrib['resolution'] == ds.attrs['resolution']:
                ULX = int(e.find('ULX').text)
                ULY = int(e.find('ULY').text)
                XDIM = int(e.find('XDIM').text)
                YDIM = int(e.find('YDIM').text)

        self.x = ULX + XDIM*np.arange(ds.totalheight)
        self.y = ULY + YDIM*np.arange(ds.totalwidth)

        self.shape = (ds.totalheight, ds.totalwidth)
        self.ndim = 2
        self.dtype = 'float32'

    def __getitem__(self, key):
        X, Y = self.x[key[1]], self.y[key[0]]
        if isinstance(key[0], slice) and isinstance(key[1], slice):
            # keys are both slices
            X, Y = np.meshgrid(X, Y)
        else:
            X, Y = np.broadcast_arrays(X, Y)

        lon, lat = self.proj(X, Y, inverse=True)

        if self.kind == 'lat':
            if hasattr(lat, 'astype'):
                return lat.astype(self.dtype)
            else:
                return np.array(lat, dtype=self.dtype)
        else:
            if hasattr(lon, 'astype'):
                return lon.astype(self.dtype)
            else:
                return np.array(lon, dtype=self.dtype)
