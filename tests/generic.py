#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Generic tests implementation
"""

from tempfile import TemporaryDirectory

from .conftest import savefig
from matplotlib import pyplot as plt
from core.tests.graphics import xrimshow, downsample
from eoread.flags import FlagsInit, GenericFlags
from datetime import datetime
from pathlib import Path
from dask import config
from core.interpolate import align_lists

import pytest
import numpy as np
import xarray as xr

from core import log
from core.tools import drop_unused_dims
from core.files import to_netcdf
from core.monitor import Chrono
from core.geo.naming import names


@pytest.fixture(params=[
    'single-threaded',
    'threads',
    # 'processes',
])
def scheduler(request):
    return request.param

@pytest.fixture(params=[
    (20, 20),                                   # two ints
    (slice(120, 130), slice(122, 135)),         # two slices
    (20, slice(120, 130)),                      # int and slice
    (slice(510, 530, 3), slice(-25, -20, 3)),   # two slices with steps
])
def indices(request):
    return request.param


class Test:
    
    @staticmethod
    def main(ds, angle_data):

        # test chunks consistency
        ds.chunks

        # Test dimensions alignment
        align_lists([ds[x].dims for x in ds])

        # check spatial dimensions
        if str(names.rows) not in ds.dims:
            raise ValueError(f'Strange dimensions, got {list(ds.dims)}')
        if str(names.columns) not in ds.dims:
            raise ValueError(f'Strange dimensions, got {list(ds.dims)}')
        
        # Check variables
        check_vars(ds)
        
        # Check band dimensions have correct coordinates
        str_types = [str] + [f'<U{i}' for i in range(1,33)]
        if ds[str(names.bands)].dtype not in str_types:
            raise ValueError(f'Coordinates of {str(names.bands)} should be bands names')
        if str(names.bgroup) not in ds: 
            raise ValueError(f'Coordinate {str(names.bgroup)} is missing')
        if ds[str(names.bgroup)].dtype not in str_types:
            raise ValueError(f'Coordinates of {str(names.bands)} should be band groups')
        if len(ds[str(names.bgroup)].dims) != 1 or ds[str(names.bgroup)].dims[0] != str(names.bands):
            raise ValueError(f'{str(names.bgroup)} should be indexed by {str(names.bands)}')
        
        # check spectral dimensions
        if str(names.ltoa) in ds and str(names.bands) not in ds[str(names.ltoa)].dims:
            raise ValueError(f'{str(names.ltoa)} variable should have a dimension called {str(names.bands)}')
            
        if str(names.rtoa) in ds and str(names.bands) not in ds[str(names.rtoa)].dims:
            raise ValueError(f'{str(names.rtoa)} variable should have a dimension called {str(names.bands)}')
            
        if str(names.bt) in ds and str(names.bands) not in ds[str(names.bt)].dims:
            raise ValueError(f'{str(names.bt)} variable should have a dimension called {str(names.bands)}')
        
        # Check dimensions order for spatial raster
        band_order = [str(names.rows), str(names.columns)]
        for name in [str(names.lat), str(names.lon), str(names.rtoa), str(names.ltoa), str(names.bt)]:
            if name not in ds: continue
            dims = list(ds[name].dims)
            if len(dims) == 2 and list(dims) != band_order:
                raise ValueError(f'Wrong order in {name}, got {dims}')
            if len(dims) == 3 and list(dims) != [str(names.bands)]+band_order:
                raise ValueError(f'Wrong order in {name}, got {dims}')
        
        # Check that following variables exist
        for name, longname in [
                (str(names.cwav), names.cwav.desc),
                (str(names.lat), names.lat.desc),
                (str(names.lon), names.lon.desc),
            ]:
            if name not in ds: 
                raise ValueError(f'Dataset should contain a variable for {longname} named {name}')

        # check that attributes exist and its type
        for name, longname, types in [
                (str(names.datetime), names.datetime.desc, str),
                (str(names.platform), names.platform.desc, str),
                (str(names.sensor), names.sensor.desc, str),            
                (str(names.resolution), names.resolution.desc, int|None),
                (str(names.product_name), names.product_name.desc, str), 
                (str(names.input_directory), names.input_directory.desc, str),
            ]:
            if name not in ds.attrs:
                raise ValueError(f'Dataset should contain a attribute named {name}')
            if not isinstance(ds.attrs[name], types):
                raise ValueError(f'Wrong type for attribute {name}')
        
        # Check that datetime is in isoformat
        try: 
            datetime.fromisoformat(ds.attrs[str(names.datetime)])
        except: 
            log.error('Issue to read the format of datetime. '
                    f'Should be isoformat, got {ds.attrs[str(names.datetime)]}')

        # Check that footprints are 2-dimensional 
        dims2 = (str(names.rows), str(names.columns))
        if ds[str(names.lat)].dims != dims2:
            raise ValueError('Latitude should be 2-dimensional')
        if ds[str(names.lon)].dims != dims2:
            raise ValueError('Longitude should be 2-dimensional')
        
        # test angle data
        if angle_data:
            
            for name, longname in [
                    (str(names.vaa), names.vaa.desc),
                    (str(names.vza), names.vza.desc),
                    (str(names.saa), names.saa.desc),
                    (str(names.sza), names.sza.desc),
                ]:
                if name not in ds: 
                    raise ValueError(f'Dataset should contain a variable for {longname} named {name}')

    @staticmethod
    def plot(request, ds, index_band):
        if str(names.ltoa) in ds: var = str(names.ltoa)
        elif str(names.rtoa) in ds: var = str(names.rtoa)
        elif str(names.bt) in ds: var = str(names.bt)
        else: raise ValueError
        
        ds[var].isel({str(names.bands): index_band}).plot.imshow()
        savefig(request)
        
    @staticmethod
    def execution_time(reader_fn, params: dict):
        with Chrono('Reading operation', unit='s'): reader_fn(**params)

    @staticmethod
    def subset(ds):
        sub = ds.isel({
            str(names.rows): slice(300, 400),
            str(names.columns): slice(500, 570)}
        )

        with TemporaryDirectory() as tmpdir,\
                config.set(scheduler='single-threaded'):
            target = Path(tmpdir)/'test.nc'
            to_netcdf(sub, target, clean_attrs=True)
    
    @staticmethod
    def lazy_load(ds):
        
        specifics = [str(names.bnames), str(names.cwav)]
        
        # Check that variables are lazy-loaded except tie_points and some specific variables
        for key, variable in ds.variables.items():
            if 'tie' in key: continue
            if key not in ds.data_vars or key in specifics: continue
            assert variable.chunks is not None, f'{key} is not lazy-loaded'
    
    @staticmethod
    def latlon(ds, flat_proj: bool = True):
        assert str(names.lat) in ds and str(names.lon) in ds
        
        # Check that latlon vary in the right way
        col = ds[str(names.lat)].isel({names.columns: 0}).values
        if col[0] > col[-1]:
            raise ValueError('Latitude is oriented to the south. Please flip it')

        row = ds[str(names.lon)].isel({names.rows: 0}).values
        if row[0] > row[-1]:
            raise ValueError('Longitude is oriented to the west. Please flip it')
        
        # Check that rasters are invariant according to one axis
        if flat_proj:
            msg = ''
            if ds[str(names.lat)].isel({names.rows: 0}).std().values > 1e-7:
                msg += 'Latitude raster has strange variation. '
                if ds[str(names.lat)].isel({names.columns: 0}).std().values < 1e-7:
                    msg += 'You probably swap dimensions (use transpose)'
                raise AssertionError(msg)
            
            if ds[str(names.lon)].isel({names.columns: 0}).std().values > 1e-7:
                msg += 'Longitude raster has strange variation. '
                if ds[str(names.lon)].isel({names.rows: 0}).std().values < 1e-7:
                    msg += 'You probably swap dimensions (use transpose)'
                raise AssertionError(msg)
    
    @staticmethod
    def angles(ds):
        angles = [names.vaa, names.vza, names.saa, names.sza]
        assert all(str(angle) in ds for angle in angles)
    
    @staticmethod
    def flagreader(ds):
        assert '_flag_reader' in ds.attrs
        flags = {GenericFlags.L1_INVALID: 4}
        flags = FlagsInit(flags, 'uint8', ds.attrs['_flag_reader'])
        result = flags.map_blocks(ds)
        assert 'flags' in result


def plot(
    request, l1: xr.Dataset, band_nir, poi: dict | None = None, yincrease: bool = False
):
    """
    Plot typical level 1 parameters to give an overview of the product.

    Args:
        yincrease: If False (default), origin is at top-left (image convention).
                   If True, origin is at bottom-left (cartographic convention).
    """
    rasters = [str(names.rtoa), str(names.bt), str(names.ltoa)]
    toa = next((v for v in rasters if v in l1), None)
    
    variables = [str(names.sza), str(names.vza), str(names.lat), str(names.lon)]
    variables = [l1[v] for v in variables if v in l1]
    for da in variables + [l1[toa].sel(bands=band_nir)]:
        data = da.compute()
        print(data.name, data.attrs)
        
        _, ax, _ = xrimshow(downsample(data), yincrease=yincrease)

        # Show point of interest
        if poi is not None:
            ax.plot(
                data.coords[data.dims[1]][poi[data.dims[1]]].values,
                data.coords[data.dims[0]][poi[data.dims[0]]].values,
                "r+",
                markersize=7,
                markeredgewidth=1,
            )
            print(data.name, ':', data.isel(poi).values.item())

        savefig(request)
    
    # Plot spectrum over the point of interest
    if poi is not None:
        l1[toa].isel(poi).plot(figsize=(5, 3))
        plt.grid(True)
        plt.title(toa)
        plt.tight_layout()
        plt.axis(ymin=0.)
        savefig(request)

def compare_version(v2, v1):
    
    v1 = drop_unused_dims(v1)
    
    # Check dimensions
    for d in v1.dims:
        assert d in v2.dims, f'[Different dimensions] {d} not in v2 : {v2.dims}'
    
    # Check coordinates
    for c, val in v1.coords.items():
        assert val.data in v2.coords[c].data, \
        f'[Different Coordinate] {c} has different values in v2 : {v2.coords[c].data} | v1 : {val.data}'
        
    # Check variables
    for v, val in v1.variables.items():
        assert v in v2.variables, f'[Different variables] {v} not in v2 : {list(v2.variables)}'
        assert val.sizes == v2.variables[v].sizes
    
    # Check attributes
    for key, val in v1.attrs.items():
        if key == 'git_commit': continue
        assert key in v2.attrs
        if isinstance(val, np.ndarray): 
            if any(val != v2.attrs[key]):
                log.warning(f'[Different values] For {key}, should be {val} but '
                            f'got {v2.attrs[key]}')            
        else:
            if val != v2.attrs[key]:
                log.warning(f'[Different values] For {key}, should be {val} but '
                            f'got {v2.attrs[key]}')


def check_vars(ds):
    
    # Check that at least one spectral variable is present
    if str(names.ltoa) not in ds and str(names.rtoa) not in ds and str(names.bt) not in ds: 
        raise ValueError(
            'No acquisitions stored in Dataset. Output Dataset should '
            f'contain at least {str(names.ltoa)} or {str(names.rtoa)} or {str(names.bt)}'
        )
    
    # Check that each variable has a unit attribute ('units' (CF) or 'unit')
    for name in [str(names.ltoa), str(names.rtoa), str(names.bt)]:
        if name not in ds:
            continue
        if not any(k in ds[name].attrs for k in ['units', 'unit']):
            raise ValueError(f'{name} does not have a unit field')
    
    # Check spectral variables values