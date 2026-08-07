#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import xarray as xr
import dask.array as da
import rasterio
import pyproj

from os import system
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

from eoread.flags import GenericFlags, FlagsReaderBase
from eoread.tools import (
    filter_metadata,
    collect_sample,
    format_chunks,
    open_raster
)

from core import log
from core.tools import merge, drop_unused_dims, only
from core.geo.naming import names
from core.table import read_xml


band_names = [
    "Band 1 - Coastal / Aerosol",
    "Band 2 - Blue",
    "Band 3 - Green",
    "Band 4 - Red",
    "Band 5 - Near Infrared",
    "Band 6 - Short Wavelength Infrared",
    "Band 7 - Short Wavelength Infrared",
    "Band 8 - Panchromatic",
    "Band 9 - Cirrus",
]

# Central wavelengths aren't described in metadata. Thus, they are hard-coded
cwvl = [442.96,482.04,561.41,654.59,864.67,1608.86,2200.73,1373.43,10895,12050,]

user_guide = 'https://greenpolicy360.net/images/Landsat8DataUsersHandbook.pdf'

def Level1_OLI(
        dirname: Union[str, Path],
        l9_angles: Union[str, Path, None] = None,
        chunks: Union[int, tuple] = 500,
        metadata_template: Union[list, None] = None,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a Landsat-8 or Landsat-9 OLI Level1 product as an xarray.Dataset.
    
    OLI (Operational Land Imager) provides 9 spectral bands from coastal aerosol
    to SWIR with 30m resolution, plus a 15m panchromatic band.

    Parameters
    ----------
    dirname : str or Path
        Path to the Landsat OLI directory
        (Example: 'LC09_L1TP_014034_20220618_20230411_02_T1/').
    l9_angles : str, Path, or None, optional
        Path to l9_angles executable for generating angle files when missing.
        The program generates sensor and solar angles with:
        `l9_angles LC0*_ANG.txt BOTH 1 -b 1`
        
        Available at: https://www.usgs.gov/land-resources/nli/landsat/
        solar-illumination-and-sensor-viewing-angle-coefficient-files
        
        Can be compiled with:
        ```
        wget https://landsat.usgs.gov/sites/default/files/documents/L9_ANGLES_2_7_0.tgz
        tar xzf L9_ANGLES_2_7_0.tgz && cd l9_angles && make
        ```
        Default is None.
    chunks : int or tuple, optional
        Size of chunks for spatial dimensions. If int, applies to both dimensions.
        Default is 500.
    metadata_template : list or None, optional
        List of metadata keys to include. If None, includes all metadata.
        Use empty list [] for minimal metadata. Default is None.
    verbose : bool, optional
        If True, print debug information. Default is True.
        
    Examples
    --------
    >>> ds = Level1_OLI('LC09_L1TP_014034_20220618_20230411_02_T1/')
    """
    
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Directory does not exists'
    
    # Format chunks
    chunks = format_chunks(chunks)

    # Read metadata
    if verbose: log.debug('read metadata')
    metadata = _Internal.read_metadata(ds, dirname, metadata_template)
    if isinstance(chunks, int): chunks = [chunks]*2

    # get datetime
    d = metadata['IMAGE_ATTRIBUTES']['DATE_ACQUIRED']
    t = metadata['IMAGE_ATTRIBUTES']['SCENE_CENTER_TIME']
    ds.attrs[str(names.datetime)] = d+'T'+t
    
    # Reading different rasters
    if verbose: log.debug('read geometric angles')
    _Internal.read_geometry(ds, dirname, l9_angles, chunks)
    if verbose: log.debug('read masks')
    _Internal.read_masks(ds, dirname, chunks)
    if verbose: log.debug('read TOA rasters')
    ds = _Internal.read_radiometry(ds, dirname, chunks)
    _Internal.read_coordinates(ds, dirname, chunks)

    # other attributes
    if verbose: log.debug('add important attributes')
    ds.attrs[str(names.platform)] = metadata['IMAGE_ATTRIBUTES']['SPACECRAFT_ID']
    ds.attrs[str(names.sensor)] = metadata['IMAGE_ATTRIBUTES']['SENSOR_ID']
    ds.attrs[str(names.product_name)] = metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']
    ds.attrs[str(names.resolution)] = 30
    ds.attrs['user_guide'] = user_guide
    ds.attrs['_flag_reader'] = 'eoread.oli.FlagsReader_OLI'

    # SRF getter
    mission = ds.attrs[str(names.platform)][-1]
    ds.attrs['_srf_getter'] = 'eoread.oli.get_srf_landsat_oli'
    ds.attrs['_srf_getter_arg'] = f'landsat_{mission}_oli'
    
    # Manage dimensions
    ds = ds.assign({str(names.cwav):((str(names.bands)), cwvl)})    
    ds = ds.rename({'y': str(names.rows), 'x': str(names.columns)})
    ds = ds.transpose(..., str(names.rows), str(names.columns))   
    ds = drop_unused_dims(ds).unify_chunks()
    ds = ds.set_coords(names.bgroup)
    
    return ds


def get_srf_landsat_oli(platform_sensor: str) -> xr.Dataset:
    """
    Read LANDSAT OLI srf, and rename its bands.
    Panchromatic bands should be properly renamed so that they can be identified
    for further processing.

    platform_sensor: "landsat_8_oli", "landsat_9_oli"
    """
    from eotools.srf import get_SRF_eumetsat, rename
    srf = get_SRF_eumetsat(platform_sensor)
    return rename(srf, band_names)


def get_sample(level: int, mission: int = 8) -> Path:
    """
    Retrieve a sample Landsat OLI product directory for testing.
    
    Returns paths to pre-configured sample products from environment variables.

    Parameters
    ----------
    level : int
        Processing level (1 for Level1, 2 for Level2).
    mission : int, optional
        Landsat mission number (8 for Landsat-8, 9 for Landsat-9).
        Default is 8.

    Returns
    -------
    Path
        Path to the Landsat OLI product directory.
        
    Raises
    ------
    ValueError
        If level is not 1 or 2.
        
    Examples
--------
    Get sample and read:

    >>> oli_dir = get_sample(level=1, mission=9)
    >>> ds = Level1_OLI(oli_dir)
    """
    collec = f'LANDSAT-{mission}-OLI'
    return collect_sample(f'LEVEL{level}_L{mission}', 'usgs', collec, level)


class FlagsReader_OLI(FlagsReaderBase):
    """
    Flags reader for OLI (Operational Land Imager) data from Landsat-8/9.
    
    Provides access to quality flags from the QA_PIXEL band for identifying
    clouds, cloud shadows, snow, and other quality indicators.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return ['QA_PIXEL']  # Use viewing zenith angle as reference
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return 'QA_PIXEL'
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the OLI dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            OLI dataset containing QA_PIXEL variable.
        flag_name : GenericFlags
            Standard flag identifier (currently only L1_INVALID supported).
        """
        if flag_name == GenericFlags.L1_INVALID:
            return ds['QA_PIXEL']
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")


################################################################################
# Intern methods
################################################################################

class _Internal:

    @staticmethod
    def read_metadata(ds: xr.Dataset, dirname: Path, template: Union[list, None]) -> dict:
        """Read and parse MTL XML metadata file."""
        filter_fn = (lambda x,y: x) if template is None else filter_metadata
        data_mtl = read_xml(only(list(dirname.glob('LC*_MTL.xml'))))
        ds.attrs['metadata'] = filter_fn(data_mtl, template)
        return data_mtl

    @staticmethod
    def read_coordinates(ds: xr.Dataset, dirname: Path, chunks: list) -> None:
        """Compute latitude and longitude arrays from UTM affine transform.

        Landsat Level-1 products are georeferenced in UTM (EPSG:326xx or
        EPSG:327xx).  Each band GeoTIFF carries an affine transform that maps
        pixel indices to UTM easting/northing.  We use this transform together
        with `pyproj` to compute accurate WGS84 lat/lon for every pixel.
        """
        # Find a VNIR band to read the transform from (B1-B5, B7 are 30 m)
        band_file = only(list(dirname.glob("LC*_B4.TIF")))

        with rasterio.open(band_file) as src:
            transform = src.transform
            crs = src.crs
            height = src.height
            width = src.width

        # Build column / row index arrays (pixel centres)
        col_indices = da.arange(width, dtype=np.float64) + 0.5
        row_indices = da.arange(height, dtype=np.float64) + 0.5

        # Apply affine transform: UTM_easting = c + a*col + b*row
        #                        UTM_northing = f + d*col + e*row
        utm_easting = transform.c + transform.a * col_indices + transform.b * (
            row_indices[:, None]
        )
        utm_northing = transform.f + transform.d * col_indices + transform.e * (
            row_indices[:, None]
        )

        # Create transformer UTM → WGS84
        utm_crs = pyproj.CRS.from_user_input(crs)
        wgs84_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

        # Transform UTM → WGS84
        # transformer.transform returns (lon, lat) for (x, y) input
        lons_dask, lats_dask = transformer.transform(
            utm_easting, utm_northing, errcheck=False
        )

        # Ensure dask arrays with proper chunks
        # chunks can be int, tuple/list, or dict like {'y': 500, 'x': 500}
        if isinstance(chunks, dict):
            chunk_tuple = (chunks.get(str(names.rows), chunks.get('y', 500)),
                          chunks.get(str(names.columns), chunks.get('x', 500)))
        elif isinstance(chunks, (list, tuple)):
            chunk_tuple = tuple(chunks)
        else:
            chunk_tuple = (chunks, chunks)

        lons_dask = da.from_array(lons_dask, chunks=chunk_tuple)
        lats_dask = da.from_array(lats_dask, chunks=chunk_tuple)

        ds[str(names.lat)] = xr.DataArray(
            lats_dask, dims=[str(names.rows), str(names.columns)],
            name=str(names.lat)
        )
        ds[str(names.lon)] = xr.DataArray(
            lons_dask, dims=[str(names.rows), str(names.columns)],
            name=str(names.lon)
        )

        ds.attrs['totalheight'] = height
        ds.attrs['totalwidth'] = width

    @staticmethod
    def gen_l9_angles(dirname: Path, l9_angles: Union[str, Path, None] = None) -> None:
        """Generate angle files using the l9_angles executable."""
        log.debug(f'Geometry file is missing in {dirname}, generating it with {l9_angles}...')
        angles_txt_file = list(dirname.glob('LC*_ANG.txt'))
        assert len(angles_txt_file) == 1, 'angle file is missing'
        assert l9_angles is not None and Path(l9_angles).exists(), \
        'Please provide a valid executable file to compute angles'
        path_exe = Path(l9_angles).absolute()
        path_angles = Path(angles_txt_file[0]).absolute()
        with TemporaryDirectory() as tmpdir:
            system(f"cd {tmpdir} && {path_exe} {path_angles} BOTH 1 -b 1")
            system(f"cp -v {tmpdir/'*'} {dirname}")

    @staticmethod
    def read_geometry(ds: xr.Dataset, dirname: Path, l9_angles: Union[str, Path, None], chunks: list) -> None:
        """Read or generate sensor and solar angle rasters."""
        # read sensor and solar angles
        for name, search in [(str(names.saa), 'LC*_SAA.TIF'),
                            (str(names.sza), 'LC*_SZA.TIF'),
                            (str(names.vaa), 'LC*_VAA.TIF'),
                            (str(names.vza), 'LC*_VZA.TIF')]:
            data = open_raster(dirname, search, engine='rasterio').chunk(chunks)
            ds[name] = (data/100).astype('float32')
        
        if (str(names.saa) not in ds) and (l9_angles is not None):
            _Internal.gen_l9_angles(dirname, l9_angles)

    @staticmethod
    def read_radiometry(ds: xr.Dataset, dirname: Path, chunks: list) -> xr.Dataset:
        """Read and calibrate radiance, reflectance, and brightness temperature."""
        rescale = ds.metadata['LEVEL1_RADIOMETRIC_RESCALING']
        thermal = ds.metadata['LEVEL1_THERMAL_CONSTANTS']
        
        # Read Panchromatic band
        dims = (str(names.columns)+'_pan', str(names.rows)+'_pan')
        files = list(dirname.glob('LC*_B8.TIF'))
        assert len(files) == 1, 'None or several files have been found for panchromatic band'
        a, m = rescale['RADIANCE_ADD_BAND_8'], rescale['RADIANCE_MULT_BAND_8']
        data = xr.open_dataarray(files[0], engine='rasterio').chunk(chunks)
        ds['Panchromatic'] = (dims,(m*data.squeeze()+a).data.astype('float32'))
        
        mus = np.cos(np.radians(ds['sza']))
        
        # Loop over bands
        for b in range(1, 12):
            # Drop Panchromatic band, it was read above
            if b == 8:
                continue
            
            # Get file for that band
            f = only(dirname.glob(f'LC*_B{b}.TIF'))

            band_str = f'_B{b}'
            
            # read radiances
            a = rescale[f'RADIANCE_ADD_BAND_{b}']
            m = rescale[f'RADIANCE_MULT_BAND_{b}']
            data = xr.open_dataarray(f, engine='rasterio').chunk(chunks)
            ds[str(names.ltoa)+band_str] = (m*data.squeeze()+a).astype('float32')
        
            # read reflectances
            if f'REFLECTANCE_ADD_BAND_{b}' not in rescale:
                ds[str(names.rtoa)+band_str] = xr.full_like(ds[str(names.ltoa)+band_str], np.nan, dtype='float32')
            else:        
                a = rescale[f'REFLECTANCE_ADD_BAND_{b}']
                m = rescale[f'REFLECTANCE_MULT_BAND_{b}']
                ds[str(names.rtoa)+band_str] = ((m*data.squeeze()+a)/mus).astype('float32')
                ds[names.bgroup+band_str] = 'bands_vnir'      
            
            # read brightness temperatures
            if f'K1_CONSTANT_BAND_{b}' not in thermal:
                ds[str(names.bt)+band_str] = xr.full_like(ds[str(names.ltoa)+band_str], np.nan, dtype='float32')
            else:        
                k1 = thermal[f'K1_CONSTANT_BAND_{b}']
                k2 = thermal[f'K2_CONSTANT_BAND_{b}']
                rad = ds[str(names.ltoa)+band_str]
                ds[str(names.bt)+band_str] = (k2/np.log(k1/rad + 1)).astype('float32')
                ds[names.bgroup+band_str] = 'bands_ir'    
            
        ds = merge(ds, dim=str(names.bands), pattern=r'(.+)_B(.+)', dtype=str)
        ds[str(names.ltoa)].attrs['unit'] = 'W/sr/m^2'
        ds[str(names.rtoa)].attrs['unit'] = None
        ds[str(names.bt)].attrs['unit'] = 'Kelvin'
        
        # Normalize x/y coordinates to integer indices (rasterio opens with UTM coords)
        # This ensures read_geometry interpolation works with consistent pixel-index coords
        ds = _Internal.normalize_spatial_coords(ds)

        return ds.sortby(ds.bands.astype(int))

    @staticmethod
    def normalize_spatial_coords(ds: xr.Dataset) -> xr.Dataset:
        """Normalize spatial dimension coordinates to integer pixel indices.
        
        rasterio opens JP2 files with UTM projected coordinates on x/y dimensions.
        This resets them to 0..N-1 so that read_geometry's interpolation (which
        uses pixel-index targets) works correctly regardless of the source coords.
        """
        return ds.assign_coords({
            str(names.columns): np.arange(ds.sizes[str(names.columns)]),
            str(names.rows): np.arange(ds.sizes[str(names.rows)]),
        })

    @staticmethod
    def read_masks(ds: xr.Dataset, dirname: Path, chunks: list) -> None:
        """Read quality assurance (QA) mask files."""
        for t in dirname.glob('*_QA_*'):
            search = re.search(r'QA_[A-Z]*', t.name)
            name = t.name[search.start():search.end()]
            ds[name] = xr.open_dataarray(t, engine='rasterio').chunk(chunks).squeeze()