#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read global surface water from https://global-surface-water.appspot.com/

https://doi.org/10.1038/nature20584

Examples
--------
Create water mask with aggregation factor 8:

>>> gsw = GSW(agg=8)

Create water mask at specific location:

>>> mask = gsw.sel(latitude=lat, longitude=lon, method='nearest') > 50
"""

import xarray as xr
import numpy as np

from pathlib import Path
from dask import array as da
from tempfile import TemporaryDirectory

from core.geo import n 
from core import env, log
from core.tools import drop_unused_dims
from core.network.download import download_url
from core.files import mdir, to_netcdf
from .common import bin_centers


from warnings import warn
warn('The module `gsw` will be deprecated from eoread package and moved to HARP package.', DeprecationWarning)


def _url_tile(tile_name):
    url = 'https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{}.tif'
    return url.format(tile_name)


class _GSW_tile:
    
    convert_missing_data = True
    
    def __init__(self, tile_name, agg, directory):
        dir_ = Path(directory).resolve()
        N = 40000/agg
        self.shape = (N, N)
        self.dtype = 'uint8'
        self.tile_name = tile_name + "v1_4_2021"
        self.agg = agg

        if not dir_.exists():
            raise IOError(
                f'Directory {dir_} does not exist. '
                'It will be used to store GSW tiles. '
                'Please create it or link it first.')

        self.filename = dir_/f'occurrence_{tile_name}_{agg}.nc'

    def __getitem__(self, key):
        
        if not self.filename.exists():
            A = xr.DataArray(
                _aggregate(
                    _fetch_gsw_tile(self.tile_name),
                    agg=self.agg),
                name='occurrence',
            )

            # set attributes
            A.attrs['aggregation factor'] = str(self.agg)
            A.attrs['source_file'] = _url_tile(self.tile_name)

            # write nc file
            to_netcdf(
                A.to_dataset(),
                filename=self.filename)
        else:
            A = xr.open_dataarray(self.filename, engine='h5netcdf', chunks={})
            
        return A[key].compute(scheduler='sync').values


def read_tile(tile_name, agg, directory):
    """
    Read a single tile as a dask array.

    Data is accessed on demand.
    """
    tile = _GSW_tile(tile_name, agg, directory)
    return da.from_array(tile, meta=np.array([], tile.dtype))


def _list_tiles():
    lons = [str(w) + "W" for w in range(180, 0, -10)]
    lons.extend([str(e) + "E" for e in range(0, 180, 10)])
    lats = [str(s) + "S" for s in range(50, 0, -10)]
    lats.extend([str(n) + "N" for n in range(0, 90, 10)])

    return lats, lons


def _aggregate(A, agg=1):
    """
    Aggregate array `A` by a factor `agg` 
    """    
    assert agg > 0, f'aggregation factor should be positive, got {agg}'
    if agg == 1: return A
    
    assert (agg & (agg-1)) == 0, f'agg should be a power of 2 ({agg})'
    return A.thin(x=agg, y=agg)


def _fetch_gsw_tile(tile_name):
    """
    Read remote file and returns its content as a numpy array
    """
    url = _url_tile(tile_name)
    
    with TemporaryDirectory() as tmpdir:
        
        # Download tiles 
        p = download_url(url, tmpdir, if_exists='skip')

        # read geotiff data
        data = xr.open_dataarray(p, engine='rasterio').squeeze().compute(scheduler='sync')
        data = data.rename(x=str(n.columns), y=str(n.rows))
        data = drop_unused_dims(data)
    
    if _GSW_tile.convert_missing_data:
        # Fill missing values
        val_nodata = 255
        data = data.where(data != val_nodata, 100)  # fill invalid data (assume water)
    
    return data

def GSW(directory=None, agg=1) -> xr.DataArray:
    """
    Global surface water reader

    Parameters
    ----------
    directory : str or Path, optional
        Directory for tile storage.
    agg : int, optional
        Aggregation factor (a power of 2). Original resolution of GSW is about
        55m at equator. Reduce this resolution by agg x agg to approximately
        match the sensor resolution. Default is 1.
            1 -> 55m
            2 -> 110m
            4 -> 220m
            8 -> 440m
            16 -> 880m

    Returns
    -------
    xr.DataArray
        Water occurrence between 0 and 100.
    """
    
    if directory is None:
        directory = mdir(env.getdir('DIR_ANCILLARY')/'GSW')

    lats, lons = _list_tiles()

    # concat the delayed dask objects for all tiles
    gsw = da.concatenate([
        da.concatenate([read_tile(f'{lon}_{lat}', agg, directory)
                        for lat in lats[::-1]], axis=0)
        for lon in lons], axis=1)

    return xr.DataArray(
        gsw,
        name='occurrence',
        dims=(str(n.lat), str(n.lon)),
        coords={
            str(n.lat): bin_centers(gsw.shape[0], 80, -60),
            str(n.lon): bin_centers(gsw.shape[1], -180, 180),
        }
    )


if __name__ == "__main__":
    # command line mode: download all GSW tiles
    # at a given aggregation factor
    import argparse
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

    log.info('Downloading GSW tiles...\n'
             '\tDirectory:', args.directory, '\n',
             '\tAggregation factor:', args.agg)

    lats, lons = _list_tiles()
    for lat in lats:
        for lon in lons:
            _GSW_tile(f'{lon}_{lat}', args.agg, args.directory)[:,:]