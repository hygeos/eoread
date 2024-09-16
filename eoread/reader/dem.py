from core import config
from core.fileutils import mdir
from eoread.utils.naming import naming
from eoread.common import bin_centers
from core.uncompress import uncompress
from eoread.download_legacy import download_url

from os.path import exists, join, basename, getsize
from os import remove, system
from math import ceil
from pathlib import Path
from dask import array as da

import xarray as xr
import numpy as np


class ArrayLike_SRTM:
    """
    Array like object to manage SRTM tiles from usgs server 
    """
    def __init__(self, directory, agg=1, missing=None, type_srtm=None, use_gdal=False, verbose=True):

        self.srtm       = 'SRTM' + str(type_srtm)
        self.agg        = agg
        self.missing    = missing
        self.use_gdal   = use_gdal
        self.verbose    = verbose
        self.directory  = Path(directory)

        static = mdir(config.get('dir_static'))
        system(f'wget https://docs.hygeos.com/s/Fy2bYLpaxGncgPM/download?files=valid_{self.srtm}_tiles.txt -c -O {static}/valid_{self.srtm}_tiles.txt')
        self.tiles_list = np.loadtxt(f'{static}/valid_{self.srtm}_tiles.txt', dtype=str)

        self.tile_size  = 3601 if type_srtm == 1 else 1201
        self.width      = 360 * self.tile_size // self.agg
        self.height     = 116 * self.tile_size // self.agg
        self.shape      = (self.height, self.width)
        self.ndim       = len(self.shape)
        self.dtype      = float

        if type_srtm == 1:
            self.url_base = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{}.SRTMGL1.hgt.zip'
        else:
            self.url_base = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11/{}.SRTMGL3.hgt.zip'

    def __getitem__(self, keys):

        ystart = int(keys[0].start) if keys[0].start is not None else 0
        xstart = int(keys[1].start) if keys[1].start is not None else 0
        ystop  = int(keys[0].stop)  if keys[0].stop is not None else self.shape[0]
        xstop  = int(keys[1].stop)  if keys[1].stop is not None else self.shape[1]

        lat_start = -int(ceil(ystop/self.tile_size - 60))
        lon_start = -int(ceil(180 - xstart/self.tile_size))
        lat_stop  = -int(ceil(ystart/self.tile_size - 60))
        lon_stop  = -int(ceil(180 - xstop/self.tile_size))
        
        # Set elements for the slicing at the end
        output_shape = (ystop-ystart,xstop-xstart)
        size_tiles = ((lat_stop-lat_start+1)*self.tile_size, (lon_stop-lon_start+1)*self.tile_size)
        alt = np.zeros(size_tiles, dtype='float32') + np.nan

        for ilon in range(lon_start, lon_stop + 1):
            for ilat in range(lat_start, lat_stop + 1):

                # Collect data tiles
                tile_name = '{}{:02d}{}{:03d}'.format(
                                        {True: 'N', False: 'S'}[ilat>=0],
                                        abs(ilat),
                                        {True: 'E', False: 'W'}[ilon>=0],
                                        abs(ilon))
                url = self.url_base.format(tile_name)
                filepath = join(self.directory,basename(url).split('.')[0]+'.hgt')
                if url in self.tiles_list:
                    if not exists(filepath):
                        filename = download_url(url, self.directory, verbose=self.verbose, wget_opts='-q')
                        filepath = uncompress(filename,self.directory)
                        remove(filename)
                    data = read_hgt(str(filepath))
                else:
                    continue
                
                # fill the concatenate matrix 
                x_origin = ilon - lon_start
                y_origin = lat_stop - ilat
                x1,x2 = x_origin*self.tile_size, (x_origin+1)*self.tile_size
                y1,y2 = y_origin*self.tile_size, (y_origin+1)*self.tile_size

                alt[y1:y2, x1:x2] = data
                alt[alt == 0] = np.nan
        
        # handling nan values
        if self.missing is None:
            assert not np.isnan(alt).any(), 'There are invalid data in SRTM'
        else: # assuming float
            alt[np.isnan(alt)] = self.missing
        assert not np.isnan(alt).any()

        # Slicing to get deserved matrix
        x_origin = xstart - xstart//self.tile_size*self.tile_size
        y_origin = ystart - ystart//self.tile_size*self.tile_size
        return alt[y_origin:y_origin+output_shape[0]:keys[0].step,x_origin:x_origin+output_shape[1]:keys[1].step]



def SRTM(directory=None, agg=1, missing=None, type_srtm=1, chunk=10000, verbose=True):
    """
    SRTM3 digital elevation model, version 2.1

    1 or 3 arc-second (~30m or ~90m) - Between 56S and 60N

    Args:
    -----

    directory: str
        directory for tile storage

    agg: int
        aggregation factor (a power of 2)
        original resolution of SRTM is about 30M or 90M at equator
        reduce this resolution by agg x agg to approximately match the sensor resolution
            1 -> 30m
            2 -> 60m
            4 -> 120m
            8 -> 240m
            16 -> 480m
    
    missing: what to provide in case of missing value
        * a float
        * None : raise an error

    type: number of arc-second to choose
        * 1 -> 30m
        * 3 -> 90m

    chunk: set size of chunks

    Returns:
    -------

    A xarray.DataArray of the DEM
    """

    assert type_srtm in [1,3]

    srtm = 'SRTM' + str(type_srtm)
    if directory is None:
        directory = mdir(config.get('dir_ancillary')/srtm)

    # concat the delayed dask objects for all tiles
    srtm = ArrayLike_SRTM(directory=directory, agg=agg, missing=missing, 
                          type_srtm=type_srtm, verbose=verbose)
    srtm = da.from_array(srtm,
                        chunks=(chunk,chunk),
                        meta=da.array([],dtype=float))

    return xr.DataArray(
        srtm,
        name='occurrence',
        dims=(naming.lat, naming.lon),
        coords={
            naming.lat: bin_centers(srtm.shape[0], 60, -56),
            naming.lon: bin_centers(srtm.shape[1], -180, 180),
        }
    )


def GTOPO30(directory=None, agg=1, missing=None, chunk=500):
    """
    GTOPO30 digital elevation model

    30 arc-second (~1km) - Between 56S and 60N

    Args:
    -----

    directory: str
        directory for tile storage

    agg: int
        aggregation factor (a power of 2)
        original resolution of GTOPO is about 1KM at equator
        reduce this resolution by agg x agg to approximately match the sensor resolution
            1 -> 1km
            2 -> 2km
            4 -> 4km
    
    missing: float to provide in case of missing value

    chunk: set size of chunks

    Returns:
    -------

    A xarray.DataArray of the DEM
    """
    
    if directory is None:
        directory = mdir(config.get('dir_ancillary')/'GTOPO30')

    # concat the delayed dask objects for all tiles
    filepath = '/archive2/data/DEM/GLOBE/GTOPO30_DZ_MLUT.nc'
    gtopo = xr.open_dataset(filepath).elev.chunk(chunks=(chunk,chunk))
    gtopo = xr.where(gtopo > 0, gtopo, missing)
    revize_dims = dict(zip(['lat','lon'], ['latitude','longitude']))
    gtopo = gtopo.rename(revize_dims)

    return gtopo.thin(agg)


def read_hgt(filename):
    """
    Reads a compressed SRTM file (binary) to a numpy array
    """
    assert filename.endswith('.hgt')
    size = getsize(filename)
    N = int(np.sqrt(size/2))
    data = np.fromfile(filename, np.dtype('>i2'), N*N)  # big endian int16
    return data.reshape((N, N))