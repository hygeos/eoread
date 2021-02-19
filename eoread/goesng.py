#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime
from pathlib import Path
from eoread.hdf4 import load_hdf4
import xarray as xr
from eoread.naming import naming
import pysolar.solar as pysol

config = {
    'auxfile': 'ANCILLARY/GOESNG-0750.1km.hdf',
}


def Level1_GOESNG(file_1km,
                  auxfile=None,
                  chunksize=1000):
    '''
    Load GOES-NG (at 1km) product as xarray Dataset

    Arguments:
        file_1km: file at 1km
            (ex: Emultic1kmNC4_goes16_201808101100.nc)
        auxfile: path to angles file (default: config['auxfile'])
    '''
    ds = xr.Dataset()

    assert 'Emultic1kmNC4' in str(file_1km)
    file_500m = Path(str(file_1km).replace('Emultic1kmNC4', 'Emultic500mNC4'))
    assert Path(file_1km).exists()
    assert file_500m.exists()

    # Load 1km data
    ds_1km = xr.open_dataset(file_1km)
    ds['VIS_004'] = ds_1km['VIS_004']
    ds['VIS_008'] = ds_1km['VIS_008']
    ds['VIS_016'] = ds_1km['VIS_016']

    # Load 500m data
    ds_500m = xr.open_dataset(file_500m, chunks=chunksize*2)
    # downsample
    arr_resampled = 0.
    for i in [0, 1]:
        for j in [0, 1]:
            arr_resampled += ds_500m.VIS_006.isel(
                nx500m=slice(i, None, 2),
                ny500m=slice(j, None, 2))
    ds['VIS_006'] = arr_resampled/4.

    # load auxiliary file
    if auxfile is None:
        auxfile = config['auxfile']

    aux = load_hdf4(auxfile)
    ds[naming.lat] = aux['Latitude']
    ds[naming.lon] = aux['Longitude']
    ds[naming.vza] = aux['View_Zenith']
    ds[naming.vaa] = aux['View_Azimuth']
    ds['Pixel_Area_Size'] = aux['Pixel_Area_Size']

    ds.attrs['auxfile'] = auxfile

    # date/time
    dt = datetime.strptime(Path(file_1km).name.split('_')[2], r'%Y%m%d%H%M%S.nc')
    ds.attrs[naming.datetime] = dt.isoformat()

    ds = ds.rename(
        Nlin=naming.rows,
        ny1km=naming.rows,
        Ncol=naming.columns,
        nx1km=naming.columns,
        ny500m=naming.rows,
        nx500m=naming.columns,
        )

    ds[naming.Rtoa] = xr.concat(
        [ds['VIS_004'], ds['VIS_006'], ds['VIS_008'], ds['VIS_016']],
        dim=naming.bands,
    )/100.

    # https://www.star.nesdis.noaa.gov/goesr/docs/ATBD/Imagery.pdf
    ds = ds.assign_coords(bands=[470, 640, 865, 1610])

    # solar angles
    ds[naming.sza] = 90.- pysol.get_altitude(
        ds.latitude, ds.longitude, dt,
        pressure=0.0, elevation=0.0,
    )
    ds[naming.saa] = pysol.get_azimuth(
        ds.latitude, ds.longitude, dt)
    
    # rechunk
    ds = ds.chunk(chunksize)

    return ds
    

