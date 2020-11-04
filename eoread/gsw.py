#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read global surface water from https://global-surface-water.appspot.com/

https://doi.org/10.1038/nature20584

Example:
-------

>>> gsw = GSW(agg=8)
Create water mask
>>> mask = gsw.sel(lat=lat, lon=lon, method='nearest') > 50
"""

import argparse
import xarray as xr
import tempfile
import numpy as np
from pathlib import Path
from dask import array as da, delayed
from urllib.request import urlopen
from urllib.error import HTTPError
from threading import Lock
from . import eo
from .common import bin_centers
from .raster import ArrayLike_GDAL

lock = Lock()

def url_tile(tile_name):
    return 'https://storage.googleapis.com/global-surface-water/downloads/occurrence/occurrence_{}.tif'.format(tile_name)


class GSW_tile:
    def __init__(self, tile_name, agg, directory, use_gdal=False):
        dir_ = Path(directory).resolve()
        N = 40000/agg
        self.shape = (N, N)
        self.dtype = 'uint8'
        self.tile_name = tile_name
        self.agg = agg
        self.use_gdal = use_gdal

        if not dir_.exists():
            raise IOError(
                f'Directory {dir_} does not exist. '
                'It will be used to store GSW tiles. '
                'Please create it or link it first.')

        self.filename = dir_/f'occurrence_{tile_name}_{agg}.nc'

    def __getitem__(self, key):
        if self.filename.exists():
            A = xr.open_dataset(self.filename, chunks={}).occurrence
        else:
            A = xr.DataArray(
                aggregate(
                    fetch_gsw_tile(self.tile_name,
                                   verbose=True,
                                   use_gdal=self.use_gdal),
                    agg=self.agg),
                dims=('height', 'width'),
                name='occurrence',
            )
            ds = A.to_dataset()

            # set attributes
            ds.attrs['aggregation factor'] = str(self.agg)
            ds.attrs['source_file'] = url_tile(self.tile_name)

            # write nc file
            eo.to_netcdf(
                ds,
                filename=self.filename)

        return A[key]


def read_tile(tile_name, agg, directory, use_gdal=False):
    '''
    Read a single tile as a dask array

    Data is accessed on demand
    '''
    tile = GSW_tile(tile_name, agg,
                    directory, use_gdal=use_gdal)
    return da.from_array(
        tile,
        meta=np.array([], tile.dtype),
    )


def list_tiles():
    lons = [str(w) + "W" for w in range(180, 0, -10)]
    lons.extend([str(e) + "E" for e in range(0, 180, 10)])
    lats = [str(s) + "S" for s in range(50, 0, -10)]
    lats.extend([str(n) + "N" for n in range(0, 90, 10)])

    return lats, lons


def aggregate(A, agg=1):
    """
    Aggregate array `A` by a factor `agg` 
    """
    assert agg > 0
    if agg == 1:
        return A

    assert agg & (agg-1) == 0, 'agg should be a power of 2 ({})'.format(agg)

    data = None
    for i in range(agg):
        for j in range(agg):
            acc = A[i::agg,j::agg]
            if data is None:
                data = acc.astype('f')
            else:
                data += acc

    return (data/(agg*agg)).astype(A.dtype)


def fetch_gsw_tile(tile_name, verbose=True, use_gdal=False):
    """
    Read remote file and returns its content as a numpy array
    """
    url = url_tile(tile_name)

    if verbose:
        print('Downloading', url)
    with tempfile.NamedTemporaryFile() as t, lock:
        # download the tile
        with urlopen(url) as response:
            raw_data = response.read()

        # write to temporary
        with open(t.name, 'wb') as fp:
            fp.write(raw_data)

        # read geotiff data
        if use_gdal:
            data = ArrayLike_GDAL(t.name)[:, :]
        else:
            data = xr.open_rasterio(t.name).isel(band=0).compute(scheduler='sync').values

    data[data == 255] = 100   # fill invalid data (assume water)

    return data


def GSW(directory='data_landmask_gsw',
        agg=1,
        use_gdal=False):
    """
    Global surface water reader

    Args:
    -----

    directory: str
        directory for tile storage

    agg: int
        aggregation factor (a power of 2)
        original resolution of GSW is about 55M at equator
        reduce this resolution by agg x agg to approximately match the sensor resolution
            1 -> 55m
            2 -> 110m
            4 -> 220m
            8 -> 440m
            16 -> 880m

    Returns:
    -------

    A xarray.DataArray of the water occurrence between 0 and 100
    """
    lats, lons = list_tiles()

    # concat the delayed dask objects for all tiles
    gsw = da.concatenate([
        da.concatenate([read_tile(f'{lon}_{lat}',
                                  agg,
                                  directory,
                                  use_gdal=use_gdal)
                        for lat in lats[::-1]], axis=0)
        for lon in lons], axis=1)


    return xr.DataArray(
        gsw,
        name='occurrence',
        dims=('lat', 'lon'),
        coords={
            'lat': bin_centers(gsw.shape[0], 80, -60),
            'lon': bin_centers(gsw.shape[1], -180, 180),
        }
    )


if __name__ == "__main__":
    # command line mode: download all GSW tiles
    # at a given aggregation factor
    parser = argparse.ArgumentParser(
        description='Download all GSW tiles at a given aggregation factor `python -m eoread.gsw`',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--directory',
                        type=str,
                        default='data_landmask_gsw',
                        help='target directory')
    parser.add_argument('--agg',
                        type=int,
                        help='aggregation factor (a power of 2)')
    args = parser.parse_args()

    if not args.agg:
        parser.print_help()
        exit()

    print('Downloading GSW tiles...')
    print('      Directory:', args.directory)
    print('      Aggregation factor:', args.agg)

    lats, lons = list_tiles()
    for lat in lats:
        for lon in lons:
            GSW_tile(
                f'{lon}_{lat}',
                args.agg,
                args.directory)[:,:]
