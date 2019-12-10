#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# TODO: check rows/columns

from os.path import dirname, exists, join

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from warnings import warn

from .common import Interpolator, DataArray_from_array
from .naming import naming, flags
from . import eo




sgli_bands = [
    380, #'VN01'
    412, #'VN02'
    443, #'VN03'
    490, #'VN04'
    530, #'VN05'
    565, #'VN06'
    673, #'VN07'
    674, #'VN08'
    763, #'VN09'
    868, #'VN10'
    869, #'VN11'
]


def Level1_SGLI(filename,
                chunks=500,
                thres_land_flag=20,
                split=False,
                ):
    """
    Read SGLI Level1 VIS-NIR bands

    Ex: GC1SG1_201912050000N02307_1BSG_VNRDK_1007.h5

    https://suzaku.eorc.jaxa.jp/GCOM_C/instruments/product.html
    """
    ds = xr.Dataset()

    # open image_data
    imdata = xr.open_dataset(
        filename,
        group='Image_data',
        chunks=chunks)

    imdata = imdata.rename_dims(dict(zip(
        imdata.Lt_VN01.dims,
        naming.dim2
    )))
    shp = (imdata.dims['rows'], imdata.dims['columns'])

    init_geometry(ds, filename, shp, chunks)

    eo.init_geometry(ds)

    ds = init_toa(ds, imdata, split)

    init_spectral(ds)

    ds = ds.assign_coords(bands=sgli_bands)

    #
    # Attributes
    #
    ga = xr.open_dataset(filename,
                         group='Global_attributes',
                         chunks=chunks)
    ds.attrs = ga.attrs
    ds.attrs[naming.datetime] = ga.attrs['Scene_center_time']
    ds.attrs[naming.product_name] = ga.attrs['Product_file_name']
    ds.attrs[naming.platform] = 'GCOM-C'
    ds.attrs[naming.sensor] = 'SGLI'

    #
    # Flags
    #
    ds[naming.flags] = xr.zeros_like(
        ds.vza,
        dtype=naming.flags_dtype)

    eo.raiseflag(
        ds[naming.flags],
        'LAND',
        flags['LAND'],
        imdata['Land_water_flag'] > thres_land_flag,
    )

    return ds


def init_toa(ds, imdata, split):

    for i, b in enumerate(sgli_bands):
        Rtoa = imdata[f'Lt_VN{i+1:02}']
        attrs = Rtoa.attrs
        Rtoa = (Rtoa & attrs['Mask']) * attrs['Slope_reflectance'] + attrs['Offset_reflectance']
        Rtoa /= ds.mus
        Rtoa.attrs = attrs
        ds[naming.Rtoa+f'_{b}'] = Rtoa

    if not split:
        ds = eo.merge(
            ds,
            [naming.Rtoa+f'_{b}' for b in sgli_bands],
            naming.Rtoa,
            naming.bands,
            )

    return ds


def init_geometry(ds, filename, shp, chunks):
    geom = xr.open_dataset(filename, group='Geometry_data')

    geom = geom.rename_dims(dict(zip(
        geom.Latitude.dims,
        ('rows_tie', 'columns_tie')
    )))

    ds['lat_tie'] = geom.Latitude
    ds['lon_tie'] = geom.Longitude

    ds['vza_tie'] = geom['Sensor_zenith']
    ds['vaa_tie'] = geom['Sensor_azimuth']
    ds['sza_tie'] = geom['Solar_zenith']
    ds['saa_tie'] = geom['Solar_azimuth']

    delta = 10
    for x in [x for x in ds if x.endswith('_tie')]:
        assert ds[x].Resampling_interval == delta

    # assign tiepoint coordinates
    ds['columns_tie'] = np.arange(ds.dims['columns_tie'])*delta
    ds['rows_tie'] = np.arange(ds.dims['rows_tie'])*delta

    # Create interpolated datasets
    for (name, A) in [
            (naming.lat, ds.lat_tie),
            (naming.lon, ds.lon_tie),
            (naming.vza, ds.vza_tie),
            (naming.vaa, ds.vaa_tie),
            (naming.sza, ds.sza_tie),
            (naming.saa, ds.saa_tie),
        ]:
        ds[name] = DataArray_from_array(
            Interpolator(shp, A),
            naming.dim2,
            chunks=chunks,
        )


def init_spectral(ds):
    """
    Read SRF
    """
    dir_auxdata = join(dirname(dirname(__file__)), 'auxdata', 'sgli')

    file_rsr = join(dir_auxdata, 'sgli_rsr_f_for_algorithm_201008.txt.gz')
    assert exists(file_rsr)

    rsr = pd.read_csv(
        file_rsr,
        engine='python',
        delim_whitespace=True,
        index_col=False,
    )

    rsr = rsr.rename(columns=dict(zip(
        [x for x in rsr.columns if x.startswith('WL')],
        [x.replace('RSR_', 'WL_') for x in rsr.columns if x.startswith('RSR')],
    )))

    wav_data = []

    # calculate central wavelengths
    for i, _ in enumerate(sgli_bands):
        srf = rsr[f'RSR_VN{i+1:02}']
        wav = rsr[f'WL_VN{i+1:02}']
        wav_eq = np.trapz(wav*srf)/np.trapz(srf)
        wav_data.append(wav_eq)

    ds['wav'] = xr.DataArray(
        da.from_array(wav_data),
        dims=(naming.bands),
    )


def show_all(filename):
    """
    List all content of netcdf file (utility function)
    """
    from netCDF4 import Dataset

    for grp in Dataset(filename).groups:
        print('')
        print('Current group is', grp)
        im = xr.open_dataset(
            filename,
            group=grp,
        )
        print(im)
