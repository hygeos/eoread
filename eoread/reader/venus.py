#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
List of VENµs bands:
-----------------

Band Use         Wavelength Bandwidth Resolution
B1   Atmo Correc 420nm      40nm      5m
B2   Aerosol     443nm      40nm      5m
B3   Water       490nm      40nm      5m
B4   Land        555nm      40nm      5m
B5   Vege Index  620nm      40nm      5m
B6   Image quali 620nm      40nm      5m
B7   Red Edge 1  667nm      30nm      5m
B8   Red Edge 2  702nm      24nm      5m
B9   Red Edge 3  742nm      16nm      5m
B10  Red Edge 4  782nm      16nm      5m
B11  Vege Index  865nm      40nm      5m
B12  Water vapor 910nm      20nm      5m
'''

# https://www.eoportal.org/satellite-missions/venus#vssc-ven%C2%B5s-superspectral-camera

from pathlib import Path
from typing import Optional
from lxml import objectify

import dask.array as da
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import rioxarray as rio
from eoread.download_legacy import download_url
from eoread.utils.fileutils import mdir
from core import config

from ..common import DataArray_from_array, Interpolator, Repeat
from ..utils.tools import raiseflag, merge
from ..utils.naming import flags, naming as n

venus_band_names = {
        420 : 'B1', 443 : 'B2',
        490 : 'B3', 555 : 'B4',
        620 : 'B5', 622 : 'B6',
        667 : 'B7', 702 : 'B8',
        742 : 'B9', 782 : 'B10', 
        865 : 'B11', 910 : 'B12',
        }


def Level1_VENUS(dirname,
               geometry=True,
               chunks=500,
               split=False):
    '''
    Read an Venµs Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA reflectances,
    the angles on the full grid, etc.

    Arguments:
        geometry: whether to read the geometry
        chunk: size of a single chunk
        split: whether the wavelength dependent variables should be split in multiple 2D variables
    '''
    ds, params = venus_read_header(dirname)
    quantif, geocoding, tileangles = params

    # lat-lon
    venus_read_latlon(ds, geocoding, chunks)

    # read geaometry
    if geometry:
        venus_read_geometry(ds, tileangles, chunks)

    # read TOA
    ds = venus_read_toa(ds, dirname, quantif, split, chunks)

    # flags
    venus_read_invalid_pix(ds, dirname, chunks, split, level=1)

    return ds


def Level2_VENUS(dirname,
                 chunks=500,
                 split=False):
    '''
    Read an Venµs Level2 product as an xarray.Dataset

    Arguments:
        chunk: size of a single chunk
        split: whether the wavelength dependent variables should be split in multiple 2D variables
    '''
    ds, params = venus_read_header(dirname)
    quantif, geocoding, tileangles = params

    # lat-lon
    venus_read_latlon(ds, geocoding, chunks)

    # read geaometry
    venus_read_geometry(ds, tileangles, chunks)

    # read reflectances
    ds = venus_read_rho(ds, dirname, quantif, split, chunks)

    # flags
    venus_read_invalid_pix(ds, dirname, chunks, split, level=2)

    return ds


def venus_read_header(dirname):
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists()

    # load xml file
    xmlfiles = list((dirname/'DATA').glob('*UII_ALL.xml'))
    assert len(xmlfiles) == 1
    xmlfile = xmlfiles[0]
    xmlroot = objectify.parse(str(xmlfile)).getroot()

    # load main xml file
    xmlfiles = list(dirname.glob('*MTD_ALL.xml'))
    assert len(xmlfiles) == 1
    xmlfile = xmlfiles[0]
    xmlgranule = objectify.parse(str(xmlfile)).getroot()

    radiometric_info = xmlgranule.Radiometric_Informations
    quantif = float(radiometric_info.REFLECTANCE_QUANTIFICATION_VALUE)
    resolution = int(xmlgranule.Radiometric_Informations.Spectral_Band_Informations_List.Spectral_Band_Informations.SPATIAL_RESOLUTION)
    
    # read date
    ds.attrs[n.datetime] = str(xmlgranule.Product_Characteristics.ACQUISITION_DATE)
    geocoding = xmlgranule.Geoposition_Informations
    tileangles = xmlgranule.Geometric_Informations.Angles_Grids_List

    # get platform
    ds.attrs[n.product_name] = xmlroot.Scene_Useful_Image_Informations.SCENE_ID.text
    # ds.attrs['crs'] = xmlgranule.Geoposition_Informations.Coordinate_Reference_System
    platform = xmlgranule.Product_Characteristics.PLATFORM.text
    assert platform in ['VENUS']

    # read image size for current resolution
    shape_info = xmlgranule.Geoposition_Informations.Geopositioning.Group_Geopositioning_List
    ds.attrs[n.totalheight] = int(shape_info.Group_Geopositioning.NROWS)
    ds.attrs[n.totalwidth] = int(shape_info.Group_Geopositioning.NCOLS)

    # attributes
    ds.attrs[n.crs]         = 'epsg:'+str(geocoding.Coordinate_Reference_System.Horizontal_Coordinate_System.HORIZONTAL_CS_CODE)
    ds.attrs[n.platform]    = platform
    ds.attrs[n.resolution]  = resolution
    ds.attrs[n.sensor]      = 'VENUS'
    ds.attrs[n.product_name] = dirname.name
    ds.attrs[n.input_directory] = str(dirname.parent)
    
    return ds, (quantif, geocoding, tileangles)


def venus_read_latlon(ds, geocoding, chunks):
    ds[n.lat] = DataArray_from_array(
        LATLON(geocoding, 'lat', ds),
        n.dim2,
        chunks=chunks,
    )

    ds[n.lon] = DataArray_from_array(
        LATLON(geocoding, 'lon', ds),
        n.dim2,
        chunks=chunks,
    )


def venus_read_invalid_pix(ds, granule_dir, chunks, split, level):
    ds[n.flags] = xr.zeros_like(
        ds.vza,
        dtype=n.flags_dtype)
    
    
    # Detect edges of tile
    if level == 1:
        if split: inv_pix = (ds.Rtoa_620 == 0) | (ds.Rtoa_622 == 0)        
        else: inv_pix = (ds.Rtoa.sel(bands=620) == 0) | (ds.Rtoa.sel(bands=622) == 0)
    elif level == 2:
        filenames = list((granule_dir/'MASKS').glob('*EDG_XS.tif'))
        assert len(filenames) == 1
        inv_pix = rio.open_rasterio(filenames[0], chunks=chunks).astype(bool)
    else:
        raise ValueError(f'Invalid value for level, got {level}')
    raiseflag(
        ds[n.flags],
        'L1_INVALID',
        flags['L1_INVALID'],
        inv_pix.squeeze()
        )

    # Flags cloud pixels
    if level == 1:
        filenames = list((granule_dir/'MASKS').glob('*CLD_XS.zip'))
        assert len(filenames) == 1
        filename = 'zip+file:'+str(filenames[0])
        cld = rio.open_rasterio(filename, chunks=chunks).astype(bool)
    elif level == 2:
        filenames = list((granule_dir/'MASKS').glob('*CLM_XS.tif'))
        assert len(filenames) == 1
        cld = rio.open_rasterio(filenames[0], chunks=chunks).astype(bool)
    raiseflag(
        ds[n.flags],
        'CLOUD_BASE',
        flags['CLOUD_BASE'],
        cld.squeeze()
        )


def venus_read_toa(ds, granule_dir, quantif, split, chunks):

    for k, v in venus_band_names.items():
        filenames = list(granule_dir.glob(f'*REF_{v}.tif'))
        assert len(filenames) == 1
        filename = filenames[0]

        arr = (rio.open_rasterio(
            filename,
            chunks=chunks,
        )/quantif).astype('float32')
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
            arr_resampled = arr_resampled.drop('band').chunk(chunks)
        else:
            # over-sample
            arr_resampled = DataArray_from_array(
                Repeat(arr, (int(1/yrat), int(1/xrat))),
                ('y', 'x'),
                chunks,
            )

        arr_resampled = arr_resampled.rename({
            'x': n.columns,
            'y': n.rows})

        arr_resampled.attrs['bands'] = k
        arr_resampled.attrs['band_name'] = v
        ds[n.Rtoa+f'_{k}'] = arr_resampled

    if not split:
        ds = merge(ds, dim=n.bands)

    return ds


def venus_read_rho(ds, granule_dir, quantif, split, chunks):

    for rho, name in zip(['SRE','FRE'],['rho_s','rho_f']):
        for k, v in venus_band_names.items():
            filenames = list(granule_dir.glob(f'*{rho}_{v}.tif'))
            assert len(filenames) == 1
            filename = filenames[0]

            arr = (rio.open_rasterio(
                filename,
                chunks=chunks,
            )/quantif).astype('float32')
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
                arr_resampled = arr_resampled.drop('band').chunk(chunks)
            else:
                # over-sample
                arr_resampled = DataArray_from_array(
                    Repeat(arr, (int(1/yrat), int(1/xrat))),
                    ('y', 'x'),
                    chunks,
                )

            arr_resampled = arr_resampled.rename({
                'x': n.columns,
                'y': n.rows})

            arr_resampled.attrs['bands'] = k
            arr_resampled.attrs['band_name'] = v
            ds[name+f'_{k}'] = arr_resampled

        if not split:
            ds = merge(ds, dim=n.bands)
    
    filenames = list(granule_dir.glob('*ATB_XS.tif'))
    assert len(filenames) == 1
    ds['ATB'] = rio.open_rasterio(filenames[0], chunks=chunks)

    return ds


def venus_read_geometry(ds, tileangles, chunks):

    # read solar angles at tiepoints
    sza = read_xml_block(tileangles.find('Sun_Angles_Grids').find('Zenith').find('Values_List'))
    saa = read_xml_block(tileangles.find('Sun_Angles_Grids').find('Azimuth').find('Values_List'))

    shp = (ds.totalheight, ds.totalwidth)

    # read view angles (for each band)
    vza = {}
    vaa = {}
    via_list = tileangles.find('Viewing_Incidence_Angles_Grids_List').find('Band_Viewing_Incidence_Angles_Grids_List')
    for e in via_list.find('Viewing_Incidence_Angles_Grids'):

        # read zenith angles
        data = read_xml_block(e.find('Zenith').find('Values_List'))
        bandid = int(e.attrib['detector_id'])
        if bandid not in vza:
            vza[bandid] = data
        else:
            ok = ~np.isnan(data)
            vza[bandid][ok] = data[ok]

        # read azimuth angles
        data = read_xml_block(e.find('Azimuth').find('Values_List'))
        bandid = int(e.attrib['detector_id'])
        if bandid not in vaa:
            vaa[bandid] = data
        else:
            ok = ~np.isnan(data)
            vaa[bandid][ok] = data[ok]

    # use the first band as vza and vaa
    k = sorted(vza.keys())[0]
    assert k in vaa

    # initialize the dask arrays
    for name, tie in [(n.sza, sza),
                      (n.saa, saa),
                      (n.vza, vza[k]),
                      (n.vaa, vaa[k]),
                      ]:
        da_tie = xr.DataArray(
            tie,
            dims=('tie_rows', 'tie_columns'),
            coords={'tie_rows': np.linspace(0, shp[0]-1, sza.shape[0]),
                    'tie_columns': np.linspace(0, shp[1]-1, sza.shape[1])})
        ds[name+'_tie'] = da_tie
        ds[name] = DataArray_from_array(
            Interpolator(shp, ds[name+'_tie']),
            n.dim2,
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
    An array-like to calculate the VENUS lat-lon
    '''
    def __init__(self, geocoding, kind, ds):
        self.kind = kind

        code = geocoding.Coordinate_Reference_System.Horizontal_Coordinate_System.HORIZONTAL_CS_CODE

        self.proj = pyproj.Proj('EPSG:{}'.format(code))

        # lookup position in the UTM grid
        geopos = geocoding.Geopositioning.Group_Geopositioning_List.Group_Geopositioning
        ULX = int(geopos.ULX)
        ULY = int(geopos.ULY)
        XDIM = int(geopos.XDIM)
        YDIM = int(geopos.YDIM)

        assert (XDIM%2 == 0) and (YDIM%2 == 0)
        self.x = ULX + XDIM//2 + XDIM*np.arange(ds.totalwidth)
        self.y = ULY + YDIM//2 + YDIM*np.arange(ds.totalheight)

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


def get_SRF(
    ds_in: Optional[xr.Dataset] = None, dir_data: Optional[Path] = None
) -> xr.Dataset:
    """
    Load Venµs spectral response functions (SRF)

    If ds_in is provided, the output bands are references by ds_in.bands
    """
    if dir_data is None:
        dir_data = mdir(config.get('dir_static')/'venus')

    url = 'https://labo.obs-mip.fr/wp-content-labo/uploads/sites/19/2018/09/rep6S.txt'
    srf_file = download_url(url, dir_data)
    nbands = 12
    ibands = range(1, nbands+1)
    df = pd.read_csv(
        srf_file,
        sep=None,
        names=['wav_um', *ibands])

    ds = xr.Dataset()
    ds.attrs["desc"] = 'Spectral response functions for VENµS'

    if ds_in is None:
        bids = ibands
    else:
        assert len(ds_in.bands) == nbands
        bids = ds_in.bands.values
    for i in range(nbands):
        ds[bids[i]] = xr.DataArray(
            df[ibands[i]].values,
            dims=["wav"],
            attrs={"band_info": f"VENUS band {bids[i]}"},
        )

    ds = ds.assign_coords(wav=df['wav_um'].values*1000)
    ds[n.wav].attrs["units"] = "nm"

    return ds

def get_sample():
    return