#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
NASA ancillary data provider

https://oceancolor.gsfc.nasa.gov/docs/ancillary/

(currently limited to NCEP GFS Forecast Meteorological data)
'''

from tempfile import TemporaryDirectory
from eoread.datetime_utils import closest, round_date
from eoread.nasa import nasa_download
from datetime import datetime
from pathlib import Path
from eoread.naming import naming
from eoread.uncompress import uncompress
import xarray as xr
import numpy as np


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

        dt = datetime.strptime(out.attrs['Start Time'][:-3], '%Y%j%H%M%S')
        return out.assign_coords(time=dt)


class Ancillary_NASA:
    def __init__(self,
                 directory='ANCILLARY/Meteorological/',
                 allow_forecast=True,
                 ):
        self.directory = Path(directory)
        self.met_resources = default_met_resources
        if allow_forecast:
            self.met_resources += forecast_resources
        self.oz_resources = default_oz_resources


    def download(self, dt: datetime, pattern: str):
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
            nasa_download(filename, target_dir)
        assert target.exists()

        return open_NASA(target)
    
    def get(self, dt: datetime, resources=None):
        '''
        Interpolate two brackting products at the given `dt`
        '''
        resources = resources or self.met_resources

        ds1, ds2 = None, None
        for res in resources:
            [(p1, d1), (p2, d2)] = res(dt)
            try:
                ds1 = self.download(d1, p1)
                ds2 = self.download(d2, p2)
            except ValueError:
                # when either product is not available
                pass
            else:
                break
        
        assert ds1 is not None

        concatenated = xr.concat([ds1, ds2], dim='time')
        
        interpolated = concatenated.interp(time=dt)

        # download ozone
        if naming.total_ozone not in interpolated:
            # try oz resources (recursively)
            oz = self.get(dt, self.oz_resources).interp(
                latitude=interpolated.latitude,
                longitude=interpolated.longitude,
            )
            return xr.merge([interpolated, oz])

        else:
            return interpolated
