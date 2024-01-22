#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
NASA ancillary data provider

https://oceancolor.gsfc.nasa.gov/docs/ancillary/

'''

from tempfile import TemporaryDirectory
from datetime import datetime
from pathlib import Path
from os import system, rename
import xarray as xr
import numpy as np
from .utils.datetime_utils import round_date
from .nasa import nasa_download
from .utils.naming import naming
from dateutil.parser import parse

from eoread.utils.config import load_config


# resources are a list of functions taking the date, and returning the list
# of file patterns and dates
forecast_resources = [
    lambda date: [("GMAO_FP.%Y%m%dT%H0000.MET.NRT.nc", d) for d in round_date(date, 3)],
]

default_resources = [
    lambda date: [("GMAO_MERRA2.%Y%m%dT%H0000.MET.nc", d) for d in round_date(date, 1)],
]


def wrap_lon(da, dim='longitude'):
    '''
    returns a wrapped dataarray along dimension `dim`, whereby
    the first element of dim is duplicated to the last position
    '''
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
        lat=naming.lat,
        lon=naming.lon)
    
    out[naming.horizontal_wind] = wrap_lon(np.sqrt(ds['U10M']**2 + ds['V10M']**2))
    out[naming.horizontal_wind].attrs.update({
        'units': ds['U10M'].units,
    })

    out[naming.sea_level_pressure] = wrap_lon(ds['SLP'])

    out[naming.total_column_ozone] = wrap_lon(ds['TO3'])

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
        """
        Initialize a provider for NASA ancillary data
        (https://oceancolor.gsfc.nasa.gov/docs/ancillary/)

        - directory: base directory for storing ancillary data.
          defaults to <dir_ancillary>/NASA (see config.py)
        """
        if directory is None:
            self.directory = load_config()['dir_ancillary']/'NASA'
        else:
            self.directory = Path(directory)

        self.resources = default_resources
        if allow_forecast:
            self.resources += forecast_resources
        self.offline = offline
        self.verbose = verbose

    def download(self,
                 dt: datetime,
                 pattern: str,
                 offline: bool = False):
        '''
        Download ancillary product at a given time (where product exists)
        '''

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
        '''
        Interpolate two brackting products at the given `dt`
        '''
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
        if naming.total_column_ozone not in interpolated:
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
