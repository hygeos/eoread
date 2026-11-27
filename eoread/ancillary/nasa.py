#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NASA ancillary data provider

https://oceancolor.gsfc.nasa.gov/resources/docs/ancillary/

Also hosts the NASA EarthData download/search utilities (`nasa_download`,
`nasa_download_uncompress`, `nasa_search`, `check_nasa_download`) that this
provider uses to fetch files; they live here because it is their only
remaining consumer in v2 (the v1 reader module has moved to
`eoread.readers.nasa`).
"""

import subprocess

from datetime import datetime
from pathlib import Path
import xarray as xr
import numpy as np
from hygeos_core.dates import round_date
from hygeos_core.files.uncompress import uncompress_decorator
from hygeos_core.geo.naming import names
from hygeos_core.network.download import download_url
from dateutil.parser import parse

from hygeos_core import env


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
    https://oceandata.gsfc.nasa.gov/data/download_methods/#api
    """
    query = [f'{k}={v}' for k, v in kwargs.items()]
    query += ['addurl=0', 'results_as_file=1']

    query_str = '&'.join(query)
    cmd = f'wget -q --post-data="{query_str}" -O - https://oceandata.sci.gsfc.nasa.gov/api/file_search'
    return subprocess.check_output(cmd, shell=True).decode().split()


# resources are a list of functions taking the date, and returning the list
# of file patterns and dates
# (module-level functions rather than lambdas, so that Ancillary_NASA can be
# pickled, e.g. for multiprocessing)
def resource_gmao_fp(date):
    return [("GMAO_FP.%Y%m%dT%H0000.MET.NRT.nc", d) for d in round_date(date, 3)]


def resource_gmao_merra2(date):
    return [("GMAO_MERRA2.%Y%m%dT%H0000.MET.nc", d) for d in round_date(date, 1)]


forecast_resources = [resource_gmao_fp]

default_resources = [resource_gmao_merra2]


def wrap_lon(da, dim='longitude'):
    """
    Return a wrapped DataArray along dimension `dim`, whereby
    the first element of dim is duplicated to the last position.
    """
    merid = da.isel({dim: 0})
    assert merid.longitude == -180
    return xr.concat([da, merid.assign_coords(longitude=180)], dim=dim)


def open_NASA(target: Path) -> xr.Dataset:
    """
    Open an ancillary file

    Warning: not all variables are considered
    """
    ds = xr.open_dataset(target, chunks={})

    out = xr.Dataset()
    out.attrs.update(ds.attrs)

    ds = ds.rename(
        lat=names.lat,
        lon=names.lon)
    
    out['horizontal_wind'] = wrap_lon(np.sqrt(ds['U10M']**2 + ds['V10M']**2))
    out['horizontal_wind'].attrs.update({
        'units': ds['U10M'].units,
    })

    out['sea_level_pressure'] = wrap_lon(ds['SLP'])

    out['total_column_ozone'] = wrap_lon(ds['TO3'])

    out['total_column_water_vapour'] = wrap_lon(ds['TQV'])

    dt = parse(out.attrs['time_coverage_start']).replace(tzinfo=None)
    dtend = parse(out.attrs['time_coverage_end']).replace(tzinfo=None)
    assert dtend == dt
    out.attrs['filename'] = target.name

    return out.assign_coords(time=dt)


class Ancillary_NASA:
    def __init__(self,
                 directory=None,
                 allow_forecast=True,
                 offline=False,
                 verbose=False,
                 ):
        """ Initialize a provider for NASA ancillary data

        (https://oceancolor.gsfc.nasa.gov/docs/ancillary/)


        Parameters
        ----------
        directory : base directory for storing ancillary data.
            defaults to <dir_ancillary>/NASA (see config.py)
        allow_forecast : bool, optional
            whether to allow forecast data, by default True
        offline : bool, optional
            if True, avoid all download, only rely on existing files, by default False
        verbose : bool, optional
            verbose mode
        """
        if directory is None:
            self.directory = env.getdir('DIR_ANCILLARY')/'NASA'
        else:
            self.directory = Path(directory)

        # copy, to avoid extending the module-level list
        self.resources = list(default_resources)
        if allow_forecast:
            self.resources += forecast_resources
        self.offline = offline
        self.verbose = verbose

    def download(self,
                 dt: datetime,
                 pattern: str,
                 offline: bool = False):
        """
        Download ancillary product at a given time (where product exists).
        """

        filename = dt.strftime(pattern)

        target_dir = self.directory/dt.strftime('%Y/%j/')
        target = target_dir/filename
        if not target.exists():
            if offline:
                raise FileNotFoundError(
                    f"Error, file {target} is not available and offline mode is set."
                )
            else:
                nasa_download(filename, target_dir)
            
        assert target.exists()

        return target

    def get(self, dt: datetime):
        """
        Interpolate two bracketing products at the given `dt`.
        """
        list_ds = None
        for offline in ([True] if self.offline else [True, False]):
            for res in self.resources:
                if list_ds is not None:
                    break
                try:
                    list_ds = [open_NASA(self.download(d, p, offline))
                               for (p, d) in res(dt)]
                except FileNotFoundError:
                    # when either product is not available
                    pass

        assert list_ds is not None, f'Error: no valid product was found for {dt} (offline={self.offline})'

        list_ds[-1].attrs['ancillary_file_1'] = list_ds[-1].attrs['filename']
        list_ds[1].attrs['ancillary_file_2'] = list_ds[1].attrs['filename']

        concatenated = xr.concat(
            list_ds, dim='time',
            combine_attrs='drop_conflicts')

        interpolated = concatenated.interp(time=dt)

        # download ozone
        if 'total_column_ozone' not in interpolated:
            # try oz resources (recursively)
            oz = self.get(dt, self.oz_resources).interp(
                latitude=interpolated.latitude,
                longitude=interpolated.longitude,
            )
            # oz.attrs = oz.rename({'filename_met_1': 'filename_oz_1'})
            oz = oz[['total_ozone']]
            oz.attrs.update({'filename_oz_1': oz.attrs.pop('filename_met_1')})
            oz.attrs.update({'filename_oz_2': oz.attrs.pop('filename_met_2')})
            return xr.merge([interpolated, oz],
                            combine_attrs='drop_conflicts')

        else:
            return interpolated