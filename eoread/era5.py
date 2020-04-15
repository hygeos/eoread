#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
ERA5 Ancillary data provider
'''

import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
import shutil

import xarray as xr
import cdsapi

from .common import floor_dt, ceil_dt
from . import eo


class ERA5:
    '''
    Ancillary data provider using ERA5
    https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5

    Arguments:
    ----------
    directory: str
        base directory for storing the ERA5 files
    pattern: str
        pattern for storing the ERA5 files in NetCDF format
    time_resolution: timedelta
        time resolution
    variables: list or None
        list of variables to download

    '''
    def __init__(self,
                 directory='ANCILLARY/ERA5/',
                 pattern='%Y/%m/%d/era5_%Y%m%d_%H%M%S.nc',
                 time_resolution=timedelta(hours=1),
                 offline=False,
                 variables=[
                     '10m_u_component_of_wind',
                     '10m_v_component_of_wind',
                     'surface_pressure',
                     'total_column_ozone',
                     'total_column_water_vapour',
                 ]
                 ):
        self.directory = Path(directory).resolve()
        self.pattern = pattern
        self.time_resolution = time_resolution
        self.client = cdsapi.Client()
        self.offline = offline

        self.variables = list(variables)

        if not self.directory.exists():
            raise Exception(
                f'Directory "{self.directory}" does not exist. '
                'Please create it for hosting ERA5 files.')


    def get(self, dt):
        """
        Download and initialize ERA5 date for a given date range

        dt: datetime or tuple of datetime
            if datetime, initialize with the closest dates
            if tuple of datetime (d0, d1), initialize with all dates
            bracketing (d0, d1)
        """
        delta = self.time_resolution
        if isinstance(dt, tuple):
            assert len(dt) == 2
            (d0, d1) = (floor_dt(dt[0], delta), ceil_dt(dt[1], delta))
        else:
            (d0, d1) = (floor_dt(dt, delta), ceil_dt(dt, delta))

        dates = [d0 + i*delta for i in range((d1-d0)//delta + 1)]

        concatenated = xr.concat([self.download(d) for d in dates], dim='time')

        wrapped = eo.wrap(concatenated,
                          'longitude',
                          -180, 180)

        return wrapped


    def download(self, dt):
        """
        Download ERA5 at a given time `dt` and returns the corresponding dataset

        Args:
        -----
        dt: datetime
        """
        assert dt.minute == 0
        assert dt.second == 0

        target = self.directory/dt.strftime(self.pattern)

        if not target.exists():
            print(f'Downloading {target}...')
            with TemporaryDirectory() as tmpdir:
                target_tmp = Path(tmpdir)/target.name
                self.client.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': self.variables,
                        'year':[f'{dt.year}'],
                        'month':[f'{dt.month:02}'],
                        'day':[f'{dt.day:02}'],
                        'time': f'{dt.hour:02}:00',
                        'format':'netcdf'
                    },
                    target_tmp)

                target.parent.mkdir(parents=True, exist_ok=True)

                shutil.move(target_tmp, str(target)+'.tmp')
                shutil.move(str(target)+'.tmp', target)
        else:
            print(f'Skipping {target}')

        return xr.open_dataset(target, chunks={})


def parse_date(dstring):
    return datetime.strptime(dstring, '%Y-%m-%d')


if __name__ == "__main__":
    # command line mode: download all ERA5 files
    # for a given time range d0 to d1
    parser = argparse.ArgumentParser(
        description='Download all ERA5 files for a given time range: `python -m eoread.era5...`')
    parser.add_argument('d0', type=parse_date,
                        help='start date (YYYY-MM-DD)')
    parser.add_argument('d1', type=parse_date,
                        help='stop date (YYYY-MM-DD)')
    parser.add_argument('--time_resolution', type=int,
                        default=1, help='time resolution in hours')
    args = parser.parse_args()

    res = ERA5().get((args.d0, args.d1))
    print(res)
