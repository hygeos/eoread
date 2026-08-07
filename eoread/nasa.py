#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read NASA Level1 files from MODIS, VIIRS, SeaWiFS

Use the L1C approach: L1C files are generated with SeaDAS (l2gen) to
get all radiometric correction

How to install SeaDAS OCSSW (see https://seadas.gsfc.nasa.gov/downloads/)

    ./install_ocssw --install_dir $HOME/ocssw --tag V2022.0 --seadas --modisa --seawifs --viirsn
"""

from pathlib import Path
import xarray as xr
from datetime import datetime
import subprocess

from core.files.uncompress import uncompress_decorator
from core.network.download import download_url
from core.geo.naming import names
from eoread import eo


def check_nasa_download(filename):
    """
    Verify that a NASA EarthData download completed successfully.
    
    Checks for HTML error pages that indicate authentication or download failures.
    
    Parameters
    ----------
    filename : Path or str
        Path to the downloaded file to check.
        
    Raises
    ------
    RuntimeError
        If the file contains HTML error content, indicating authentication
        failure or file not found.
        
    Notes
    -----
    Requires .netrc file with NASA EarthData credentials.
    See: https://support.earthdata.nasa.gov/index.php?/Knowledgebase/Article/View/43/21/
    """
    errormsg = 'Error authenticating to NASA EarthData for downloading ancillary data. ' \
    'Please provide authentication through .netrc. See more information on ' \
    'https://support.earthdata.nasa.gov/index.php?/Knowledgebase/Article/View/43/21/how-to-access-urs-gated-data-with-curl-and-wget'
    with open(filename, 'rb') as fp:
        filehead = fp.read(100)
        if filehead.startswith((
            b'<!DOCTYPE html>',
            # may be the case after Oct 2023 when NASA changed the APIs
            b'404 Error',
            b'403 Error')):
            raise RuntimeError(errormsg)


def nasa_download(product, dirname, tmpdir=None, verbose=True, wget_extra=""):
    """
    Download a NASA ocean color product from oceandata.sci.gsfc.nasa.gov.
    
    Supports MODIS, VIIRS, SeaWiFS, and Sentinel-3 OLCI products.
    Automatically handles authentication using wget cookie files.
    
    Parameters
    ----------
    product : str
        Product filename or full URL to download.
    dirname : str or Path
        Target directory for the downloaded file.
    tmpdir : str or Path, optional
        Temporary directory for partial downloads.
    verbose : bool, optional
        If True, prints download progress. Default is True.
    wget_extra : str, optional
        Additional wget command-line options.
        
    Returns
    -------
    Path
        Path to the downloaded file.
        
    Examples
    --------
    >>> nasa_download('A2005005002500.L1A_LAC.bz2', '/data/')
    >>> nasa_download('S3A_OL_1_EFR_*.zip', '/data/')  # Sentinel-3
    
    Notes
    -----
    Full URLs can be provided instead of just product names.
    """
    if product.startswith('https://'):
        url = product
    elif product.startswith('S3'):
        url= f'https://oceandata.sci.gsfc.nasa.gov/sentinel/getfile/{product}.zip'
    else:
        url = f'https://oceandata.sci.gsfc.nasa.gov/getfile/{product}'

    return download_url(
        url,
        dirname,
        verbose=verbose,
        tmpdir=tmpdir,
        wget_opts='-nv --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies ' \
                  '--keep-session-cookies --auth-no-challenge '+wget_extra,
        check_function=check_nasa_download,
        lock_timeout=3600,
        if_exists='skip',
        )


def nasa_download_uncompress(product, dirname) -> Path:
    """
    Download and automatically uncompress a NASA product.
    
    Combines nasa_download with automatic decompression of .bz2, .gz,
    and .zip archives.
    
    Parameters
    ----------
    product : str
        Product filename to download.
    dirname : str or Path
        Target directory.
        
    Returns
    -------
    Path
        Path to the uncompressed file.
    """
    return uncompress_decorator()(nasa_download)(product, dirname)


def nasa_search(**kwargs):
    """
    Search for NASA ocean color products on oceandata.sci.gsfc.nasa.gov.
    
    Uses the NASA OceanColor API to find products matching the search criteria.
    
    Parameters
    ----------
    **kwargs : dict
        Search parameters passed to the API, such as:
        - sensor: Sensor name ('seawifs', 'modis', 'viirs', etc.)
        - sdate: Start date (YYYY-MM-DD)
        - edate: End date (YYYY-MM-DD)
        - dtype: Data type ('L1', 'L2', 'L3', etc.)
        - search: Filename pattern
        
    Returns
    -------
    list
        List of product filenames matching the search criteria.
        
    Examples
    --------
    >>> products = nasa_search(
    ...     sensor='seawifs',
    ...     sdate='2000-04-17',
    ...     edate='2000-04-17',
    ...     dtype='L1',
    ...     search='*L1A_GAC'
    ... )
    
    See
    ---
    https://oceancolor.gsfc.nasa.gov/data/download_methods/#api
    """
    query = [f'{k}={v}' for k, v in kwargs.items()]
    query += ['addurl=0', 'results_as_file=1']

    query_str = '&'.join(query)
    cmd = f'wget -q --post-data="{query_str}" -O - https://oceandata.sci.gsfc.nasa.gov/api/file_search'
    return subprocess.check_output(cmd, shell=True).decode().split()


# Per-sensor configuration: which sensors need spatial reversal to normalize orientation
# MODIS and VIIRS store data with first line = northernmost, first pixel = easternmost
# HAWKEYE and SeaWiFS store data in standard geographic order
_SENSOR_REVERSE = {
    'MODIS': True,
    'VIIRS': True,
    'HAWKEYE': False,
    'SEAWIFS': False,
}


def Level1_NASA(filename, chunks=500, normalize_orientation=True):
    """
    Read a NASA L1C product (MODIS, VIIRS, SeaWiFS) as an xarray.Dataset.
    
    L1C products are generated with SeaDAS l2gen to include polarization
    correction and other radiometric corrections.
    
    Parameters
    ----------
    filename : Path or str
        Path to the NASA L1C NetCDF file.
    chunks : int, optional
        Chunk size for spatial dimensions. Default is 500.
    normalize_orientation : bool, optional
        If True (default), normalize the spatial orientation so that latitude
        increases from bottom to top and longitude increases from left to right.
        This is done by reversing rows and columns for sensors that store data
        in scan order (see _SENSOR_REVERSE). Set to False to get raw data in
        the file's native orientation.
        
    Returns
    -------
    xr.Dataset
        Dataset containing:
            - Rtoa: Top-of-atmosphere reflectance (polarization corrected)
            - VZA, VAA, SZA, SAA: Viewing and solar geometry
            - lat, lon: Geolocation
            - flags: Quality flags (LAND, L1_INVALID)

    Notes
    -----
    Requires SeaDAS OCSSW installation. See:
    https://seadas.gsfc.nasa.gov/downloads/
        
    Examples
    --------
    >>> ds = Level1_NASA('A2005005002500.L1C.nc', chunks=1000)
    """
    ds = xr.open_dataset(filename, chunks=chunks)
    # TODO: use xr.open_datatree instead of several xr.open_dataset

    dstart = datetime.strptime(ds.attrs['time_coverage_start'], "%Y-%m-%dT%H:%M:%S.%fZ")
    dstop = datetime.strptime(ds.attrs['time_coverage_end'], "%Y-%m-%dT%H:%M:%S.%fZ")
    d = dstart + (dstop - dstart)//2
    ds.attrs[str(names.datetime)] = d.isoformat()
    ds.attrs[str(names.sensor)] = ds.attrs['instrument']
    ds.attrs[str(names.input_directory)] = str(Path(filename).parent)
    
    # SRF getter based on platform/sensor combination
    platform = ds.attrs.get('platform', '').lower()
    instrument = ds.attrs.get('instrument', '').lower()
    ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_NASA'
    ds.attrs['_srf_getter_arg'] = f'{platform}_{instrument}'

    sensor_band = xr.open_dataset(filename, group='/sensor_band_parameters', chunks=chunks)
    bands = sensor_band['wavelength'].values[sensor_band.number_of_reflective_bands.values].astype('int32')

    # Determine whether this sensor needs reversal for normalized orientation
    reverse = False
    if normalize_orientation:
        instrument = ds.attrs.get('instrument', '').upper()
        reverse = _SENSOR_REVERSE.get(instrument, False)

    navi = xr.open_dataset(filename, group='navigation_data', chunks=chunks)
    navi = navi.rename_dims({'number_of_lines':str(names.rows), 'pixels_per_line':str(names.columns)})
    if reverse:
        navi = navi.isel({str(names.rows): slice(None, None, -1),
                          str(names.columns): slice(None, None, -1)})
    ds[str(names.lat)] = navi.latitude
    ds[str(names.lon)] = navi.longitude
    
    geo_data = xr.open_dataset(filename, group='/geophysical_data', chunks=chunks)
    geo_data = geo_data.rename_dims({'number_of_lines':str(names.rows), 'pixels_per_line':str(names.columns)})
    if reverse:
        geo_data = geo_data.isel({str(names.rows): slice(None, None, -1),
                                  str(names.columns): slice(None, None, -1)})
    for _n,r,p in [(str(names.rtoa)+f'_{b}', f'rhot_{b}', f'polcor_{b}') for b in bands]:
        try:
            ds[_n] = (geo_data[r]/geo_data[p]).where(geo_data[r] > -100.)
        except:
            pass

    for (name, param) in [(str(names.sza), 'solz'),
                          (str(names.vza), 'senz'),
                          (str(names.saa), 'sola'),
                          (str(names.vaa), 'sena'),
                          ]:
        ds[name] = geo_data[param]

    eo.init_geometry(ds)

    ds[str(names.flags)] = xr.zeros_like(ds[str(names.lat)], dtype=names.flags.dtype)
    for flag, flag_list, flag_value in [
        ("LAND", ["LAND"], 1),
        ("L1_INVALID", ["ATMFAIL", "PRODFAIL"], 4),
    ]:
        flag_mask = 0
        for f in flag_list:
            flag_mask += geo_data.l2_flags.flag_masks[geo_data.l2_flags.flag_meanings.split().index(f)]

        eo.raiseflag(
            ds[names.flags],
            flag,
            flag_value,
            (geo_data.l2_flags & flag_mask != 0),
        )

    ds = eo.merge(ds, dim=str(names.bands))
    return ds
