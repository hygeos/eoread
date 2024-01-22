#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
NASA ancillary data provider

https://oceancolor.gsfc.nasa.gov/docs/ancillary/

(currently limited to NCEP GFS Forecast Meteorological data)
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
from .utils.uncompress import uncompress
from .utils import hdf4


# resources are a list of functions taking the date, and returning the list
# of file patterns and dates
forecast_resources = [
    lambda date: [('N%Y%j%H_MET_NCEP_1440x0721_f{}.hdf'.format(
                   '012' if (d.hour % 2 == 0) else '015'), d)
                  for d in round_date(date, 3)]
]

default_met_resources = [
    lambda date: [('N%Y%j%H_MET_NCEPR2_6h.hdf.bz2', d) for d in round_date(date, 6)],
    lambda date: [('N%Y%j%H_MET_NCEP_6h.hdf.bz2', d) for d in round_date(date, 6)],
    lambda date: [('N%Y%j%H_MET_NCEP_6h.hdf', d) for d in round_date(date, 6)],
]

default_oz_resources = [
    lambda date: [('N%Y%j00_O3_AURAOMI_24h.hdf', d) for d in round_date(date, 24)],
    lambda date: [('N%Y%j00_O3_TOMSOMI_24h.hdf', d) for d in round_date(date, 24)],
    lambda date: [('S%Y%j00%j23_TOAST.OZONE', d) for d in round_date(date, 24)],
    lambda date: [('S%Y%j00%j23_TOVS.OZONE', d) for d in round_date(date, 24)],
]


def wrap(da, dim):
    '''
    returns a wrapped dataarray along dimension `dim`, whereby
    the first element of dim is duplicated to the last position
    '''
    return xr.concat([da, da.isel({dim: 0})], dim=dim)


def open_NASA(target):
    with TemporaryDirectory() as tmpdir:

        uncompressed = uncompress(
            target,
            tmpdir,
            on_uncompressed='bypass')

        if uncompressed.name.endswith('.hdf'):
            ds = hdf4.load_hdf4(uncompressed)
        else:
            ds = xr.open_dataset(uncompressed, chunks={})

        out = xr.Dataset()
        out.attrs.update(ds.attrs)

        for v in ds:
            ds[v] = ds[v].rename(dict(zip(ds[v].dims, ('latitude', 'longitude'))))
        
        nlat = ds.latitude.size
        nlon = ds.longitude.size
        assert nlat < nlon

        if ('z_wind' in ds) and ('m_wind' in ds):
            out[naming.horizontal_wind] = wrap(np.sqrt(np.sqrt(ds.z_wind**2 + ds.m_wind**2)),
                                            dim='longitude')
        if 'press' in ds:
            out[naming.sea_level_pressure] = wrap(ds.press, dim='longitude')
        if 'ozone' in ds:
            out[naming.total_ozone] = wrap(ds.ozone, dim='longitude')
        out = out.assign_coords(
            latitude=np.linspace(90, -90, nlat),
            longitude=np.linspace(-180, 180, nlon+1),
        )

        dt = datetime.strptime(out.attrs['Start Time'][:13], '%Y%j%H%M%S')
        out.attrs['filename'] = target.name

        return out.assign_coords(time=dt)


class Ancillary_NASA:
    def __init__(self,
                 directory='ANCILLARY/Meteorological/',
                 allow_forecast=True,
                 offline=False,
                 verbose=False,
                 ):
        self.directory = Path(directory)
        self.met_resources = default_met_resources
        self.oz_resources = default_oz_resources
        if allow_forecast:
            self.met_resources += forecast_resources
            self.oz_resources += forecast_resources
        self.offline = offline
        self.verbose = verbose

    def download(self, dt: datetime, pattern: str,
                 offline: bool = False):
        '''
        Download ancillary product at a given time (where product exists)
        '''
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.hour % 3 == 0

        filename = dt.strftime(pattern)

        target_dir = self.directory/dt.strftime('%Y/%j/')
        target = target_dir/filename
        if not target.exists():
            if offline:
                if self.verbose:
                    print(f'Trying file {target} : NOT FOUND [OFFLINE]')
                raise FileNotFoundError(
                    f'Error, file {target} is not available and offline mode is set.')
            else:
                try:
                    nasa_download(filename, target_dir)
                except FileNotFoundError:
                    if self.verbose:
                        print(f'Trying file {target} : NOT FOUND [ONLINE]')
                except RuntimeError:
                    if self.verbose:
                        print('Authentification issue : Please check your login in .netrc')
                    raise
                if self.verbose:
                    print(f'Trying file {target} : FOUND [ONLINE]')
        else:
            if self.verbose:
                print(f'Trying file {target} : FOUND [OFFLINE]')
            
        assert target.exists()
        target = verify(target)

        return open_NASA(target)

    def get(self, dt: datetime, resources=None):
        '''
        Interpolate two brackting products at the given `dt`
        '''
        resources = resources or self.met_resources

        list_ds = None
        for offline in ([True] if self.offline else [True, False]):
            for res in resources:
                if list_ds is not None:
                    break
                try:
                    list_ds = [self.download(d, p, offline)
                               for (p, d) in res(dt)]
                except FileNotFoundError:
                    # when either product is not available
                    pass

        assert list_ds is not None, f'Error: no valid product was found for {dt} (offline={self.offline})'

        list_ds[0].attrs['filename_met_1'] = list_ds[0].attrs['filename']
        list_ds[1].attrs['filename_met_2'] = list_ds[1].attrs['filename']

        concatenated = xr.concat(
            list_ds, dim='time',
            combine_attrs='drop_conflicts')

        if len(list_ds) == 1:
            interpolated = concatenated
        else:
            interpolated = concatenated.interp(time=dt)

        # download ozone
        if naming.total_ozone not in interpolated:
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


def verify(filename):
    '''
    Fix files with wrong extension from NASA
    -> HDF files with bz2 extension
    '''
    if filename.name.endswith('.bz2') and system(f'bzip2 -t {filename}'):
        target = filename.parent/filename.name[:-4]
        rename(filename, target)
        filename = target

    return filename

