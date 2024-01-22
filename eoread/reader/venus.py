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
from lxml import objectify

import dask.array as da
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import rioxarray as rio
from eoread.download_legacy import download_url

from ..common import DataArray_from_array, Interpolator, Repeat
from ..utils.tools import raiseflag, merge
from ..utils.naming import naming, flags

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
    Read an VENµs Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA reflectances,
    the angles on the full grid, etc.

    Arguments:
        resolution: '5' (in m)
        geometry: whether to read the geometry
        split: whether the wavelength dependent variables should be split in multiple 2D variables
    '''
    ds = xr.Dataset()
    dirname = Path(dirname).resolve()

    # load xml file
    xmlfiles = list((dirname/'DATA').glob('*.xml'))
    assert len(xmlfiles) == 1
    xmlfile = xmlfiles[0]
    xmlroot = objectify.parse(str(xmlfile)).getroot()

    # load main xml file
    xmlfiles = list(dirname.glob('*.xml'))
    assert len(xmlfiles) == 1
    xmlfile = xmlfiles[0]
    xmlgranule = objectify.parse(str(xmlfile)).getroot()

    radiometric_info = xmlgranule.Radiometric_Informations
    quantif = float(radiometric_info.REFLECTANCE_QUANTIFICATION_VALUE)
    resolution = int(xmlgranule.Radiometric_Informations.Spectral_Band_Informations_List.Spectral_Band_Informations.SPATIAL_RESOLUTION)
    # read date
    ds.attrs['datetime'] = str(xmlgranule.Product_Characteristics.ACQUISITION_DATE)
    geocoding = xmlgranule.Geoposition_Informations
    tileangles = xmlgranule.Geometric_Informations.Angles_Grids_List

    # get platform
    ds.attrs['tile_id'] = xmlroot.Scene_Useful_Image_Informations.SCENE_ID.text
    # ds.attrs['crs'] = xmlgranule.Geoposition_Informations.Coordinate_Reference_System
    platform = xmlgranule.Product_Characteristics.PLATFORM.text
    assert platform in ['VENUS']

    # read image size for current resolution
    shape_info = xmlgranule.Geoposition_Informations.Geopositioning.Group_Geopositioning_List
    ds.attrs[naming.totalheight] = int(shape_info.Group_Geopositioning.NROWS)
    ds.attrs[naming.totalwidth] = int(shape_info.Group_Geopositioning.NCOLS)

    # attributes
    ds.attrs[naming.platform] = platform
    ds.attrs['resolution'] = resolution
    ds.attrs[naming.sensor] = 'VENUS'
    ds.attrs[naming.product_name] = dirname.name
    ds.attrs[naming.input_directory] = str(dirname.parent)

    # lat-lon
    venus_read_latlon(ds, geocoding, chunks)

    # venus_read_geometry
    if geometry:
        venus_read_geometry(ds, tileangles, chunks)

    # venus_read_toa
    ds = venus_read_toa(ds, dirname, quantif, split, chunks)

    # read spectral information
    venus_read_spectral(ds, radiometric_info)

    # flags
    ds[naming.flags] = xr.zeros_like(
        ds.vza,
        dtype=naming.flags_dtype)
    raiseflag(
        ds[naming.flags],
        'L1_INVALID',
        flags['L1_INVALID'],
        np.isnan(ds.vza)
        )

    return ds


def venus_read_latlon(ds, geocoding, chunks):
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
            'x': naming.columns,
            'y': naming.rows})

        arr_resampled.attrs['bands'] = k
        arr_resampled.attrs['band_name'] = v
        ds[naming.Rtoa+f'_{k}'] = arr_resampled

    if not split:
        ds = merge(ds, dim=naming.bands)

    return ds


def venus_read_spectral(ds, radiometric_info):
    wav_data = []

    for s in radiometric_info.Spectral_Band_Informations_List.Spectral_Band_Informations:
        wav_eq = int(s.Wavelength.CENTRAL)
        wav_data.append(wav_eq)

    ds['wav'] = xr.DataArray(
        da.from_array(wav_data),
        dims=(naming.bands),
    ).chunk({naming.bands: 1})


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

        code = geocoding.Coordinate_Reference_System.Horizontal_Coordinate_System.HORIZONTAL_CS_CODE

        self.proj = pyproj.Proj('EPSG:{}'.format(code))

        # lookup position in the UTM grid
        geopos = geocoding.Geopositioning.Group_Geopositioning_List.Group_Geopositioning
        ULX = int(geopos.ULX)
        ULY = int(geopos.ULY)
        XDIM = int(geopos.XDIM)
        YDIM = int(geopos.YDIM)

        assert (XDIM%2 == 0) and (YDIM%2 == 0)
        self.x = ULX + XDIM//2 + XDIM*np.arange(ds.totalheight)
        self.y = ULY + YDIM//2 + YDIM*np.arange(ds.totalwidth)

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


def get_SRF() -> xr.Dataset:
    """
    Load Venµs spectral response functions (SRF)
    """
    dir_data = mdir(naming.dir_static/'venus')
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

    for bid in ibands:
        ds[bid] = xr.DataArray(
            df[bid].values,
            dims=["wav"],
            attrs={"band_info": f"VENUS band {bid}"},
        )
    
    ds = ds.assign_coords(wav=df['wav_um'].values*1000)
    ds['wav'].attrs["units"] = "nm"

    return ds