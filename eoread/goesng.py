#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from eoread.hdf4 import load_hdf4
import xarray as xr
from eoread.naming import naming
from eoread.common import Repeat, DataArray_from_array

config = {
    'auxfile': 'ANCILLARY/GOESNG-0750.1km.hdf',
}


def Level1_GOESNG(file_1km,
                  auxfile=None,
                  chunksize=500):
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

    # rechunk
    ds = ds.rename(
        Nlin=naming.rows,
        ny1km=naming.rows,
        Ncol=naming.columns,
        nx1km=naming.columns,
        ny500m=naming.rows,
        nx500m=naming.columns,
        )
    
    ds = ds.chunk(chunksize)

    return ds
    

