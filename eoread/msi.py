#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pyproj
import xarray as xr
import dask.array as da
from glob import glob
from lxml import objectify
from datetime import datetime
import os
from eoread.common import rectBivariateSpline, Repeat

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


msi_band_names = {
        443 : 'B01', 490 : 'B02',
        560 : 'B03', 665 : 'B04',
        705 : 'B05', 740 : 'B06',
        783 : 'B07', 842 : 'B08',
        865 : 'B8A', 940 : 'B09',
        1375: 'B10', 1610: 'B11',
        2190: 'B12',
        }


def Level1_MSI(dirname, resolution='60', geometry=True):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances, the angles on the full grid, etc.

    resolution: '60', '20' or '10' (in m)
    '''
    ds = xr.Dataset()

    if dirname.endswith('.SAFE') or dirname.endswith('.SAFE/'):
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
    ds.attrs['datetime'] = datetime.strptime(str(xmlgranule.General_Info.find('SENSING_TIME')),
                                            '%Y-%m-%dT%H:%M:%S.%fZ')
    geocoding = xmlgranule.Geometric_Info.find('Tile_Geocoding')
    tileangles = xmlgranule.Geometric_Info.find('Tile_Angles')

    # get platform
    tile_id = str(xmlgranule.General_Info.find('TILE_ID')[0])
    platform = tile_id[:3]
    assert platform in ['S2A', 'S2B']

    # read image size for current resolution
    for e in geocoding.findall('Size'):
        if e.attrib['resolution'] == str(resolution):
            ds['totalheight'] = int(e.find('NROWS').text)   # TODO: move to attributes
            ds['totalwidth'] = int(e.find('NCOLS').text)
            break

    # attributes
    ds.attrs['platform'] = platform
    ds.attrs['resolution'] = resolution

    # lat-lon
    msi_read_latlon(ds, geocoding)

    # msi_read_geometry
    if geometry:
        msi_read_geometry(ds, tileangles)

    # msi_read_toa
    msi_read_toa(ds, granule_dir, quantif)

    # read spectral information
    msi_read_spectral(ds)

    return ds


def msi_read_latlon(ds, geocoding):

    code = geocoding.find('HORIZONTAL_CS_CODE').text

    print('Initialize MSI projection {}'.format(code))

    proj = pyproj.Proj('+init={}'.format(code))

    # lookup position in the UTM grid
    for e in geocoding.findall('Geoposition'):
        if e.attrib['resolution'] == ds.attrs['resolution']:
            ULX = int(e.find('ULX').text)
            ULY = int(e.find('ULY').text)
            XDIM = int(e.find('XDIM').text)
            YDIM = int(e.find('YDIM').text)

    X, Y = np.meshgrid(ULX + XDIM*np.arange(ds['totalheight']), 
                       ULY + YDIM*np.arange(ds['totalwidth']))

    lon, lat = (proj(X, Y, inverse=True))

    # shp = (ds['totalheight'], ds['totalwidth'])
    shp = ('columns', 'rows')

    ds['latitude'] = (shp, lat)
    ds['longitude'] = (shp, lon)


def msi_read_toa(ds, granule_dir, quantif):
    chunks = {'x': 400,
              'y': 300}

    for k, v in msi_band_names.items():
        filenames = glob(os.path.join(granule_dir, 'IMG_DATA', f'*_{v}.jp2'))
        assert len(filenames) == 1
        filename = filenames[0]

        arr = xr.open_rasterio(filename, chunks=chunks)/quantif
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
        else:
            # over-sample
            arr_resampled = (('columns', 'rows'),
                             da.from_array(Repeat(arr, (int(1/xrat), int(1/yrat)))))

        # da = da.rename({'x': f'x_{v}',
        #                 'y': f'y_{v}'})
        # ds[f'Rtoa_{k}'] = da
        # TODO: rows and columns may be mixed up
        arr_resampled = arr_resampled.rename({
                            'x': 'rows',
                            'y': 'columns'})
        ds[f'Rtoa_{k}'] = arr_resampled

def msi_read_spectral(ds):
    pass

def msi_read_geometry(ds, tileangles):

    # read solar angles at tiepoints
    sza = read_xml_block(tileangles.find('Sun_Angles_Grid').find('Zenith').find('Values_List'))
    saa = read_xml_block(tileangles.find('Sun_Angles_Grid').find('Azimuth').find('Values_List'))

    shp = (int(ds.totalheight), int(ds.totalwidth))
    shpn = ('columns', 'rows')

    ds['sza'] = (shpn, rectBivariateSpline(sza, shp))
    ds['saa'] = (shpn, rectBivariateSpline(saa, shp))

    # read view angles (for each band)
    vza = {}
    vaa = {}
    for e in tileangles.findall('Viewing_Incidence_Angles_Grids'):

        # read zenith angles
        data = read_xml_block(e.find('Zenith').find('Values_List'))
        bandid = int(e.attrib['bandId'])
        if not bandid in vza:
            vza[bandid] = data
        else:
            ok = ~np.isnan(data)
            vza[bandid][ok] = data[ok]

        # read azimuth angles
        data = read_xml_block(e.find('Azimuth').find('Values_List'))
        bandid = int(e.attrib['bandId'])
        if not bandid in vaa:
            vaa[bandid] = data
        else:
            ok = ~np.isnan(data)
            vaa[bandid][ok] = data[ok]

    ds['vza'] = (shpn, np.zeros(shp, dtype='float32'))
    ds['vaa'] = (shpn, np.zeros(shp, dtype='float32'))

    # use the first band as vza and vaa
    k = sorted(vza.keys())[0]
    assert k in vaa
    ds.vza[:,:] = rectBivariateSpline(vza[k], shp)
    ds.vaa[:,:] = rectBivariateSpline(vaa[k], shp)


def read_xml_block(item):
    '''
    read a block of xml data and returns it as a numpy float32 array
    '''
    d = []
    for i in item.iterchildren():
        d.append(i.text.split())
    return np.array(d, dtype='float32')

