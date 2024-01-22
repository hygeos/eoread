#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime as dt
from matplotlib import pyplot as plt

import pytest

from eoread.era5 import ERA5
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.reader.meris import Level1_MERIS
from eoread.sample_products import get_sample_products
from eoread.utils.datetime_utils import closest
from eoread.utils.tools import datetime
from . import conftest

p = get_sample_products()



@pytest.fixture(params=[
    'total_ozone',
    'sea_level_pressure',
    'horizontal_wind'])
def variable(request):
    return request.param

@pytest.mark.parametrize('ancillary,args', [
        (ERA5, (dt(2010, 1, 1, 1, 0, 0),)),
        (Ancillary_NASA, (closest(dt.now(), 6), 'N%Y%j%H_MET_NCEP_1440x0721_f012.hdf')),
        ])
def test_download(request, args, ancillary, variable):
    """
    Download a single file
    """
    anc = ancillary().download(*args)

    # check that it is properly wrapped
    assert anc.latitude.min() == -90
    assert anc.latitude.max() == 90
    assert anc.longitude.min() == -180
    assert anc.longitude.max() == 180
    assert (anc[variable].isel(longitude=0) == anc[variable].isel(longitude=-1)).all()
    assert 'time' not in anc.dims

    anc[variable].plot()
    plt.title(variable)
    conftest.savefig(request)


@pytest.mark.parametrize('ancillary,date', [
        (ERA5, dt(2010, 1, 1, 1, 30, 0)),
        (Ancillary_NASA, dt.now()),
        (Ancillary_NASA, dt(2010, 1, 1, 0, 0, 0)),
        ])
def test_get_single(request, date, ancillary, variable):
    """
    Download and interpolate between bracketing files
    """
    anc = ancillary().get(date)
    assert not 'time' in anc.dims
    anc[variable].plot()
    plt.title(variable)
    conftest.savefig(request)


@pytest.mark.parametrize('ancillary', [ERA5, Ancillary_NASA])
@pytest.mark.parametrize('method', ['nearest', 'linear'])
def test_interp(request, method, ancillary, variable):
    """
    Fetch and interpolate ancillary data for a MERIS product

    We test with an orbit that crosses the date line
    """
    def get_unit(da):
        if 'units' in da.attrs:
            return da.units
        elif 'unit' in da.attrs:
            return da.unit
        else:
            return '<unit not available>'
        
    l1 = Level1_MERIS(p['prod_meris_L1_20080701']['path']).sel(
        y=slice(2000, 2600),
        )

    anc = ancillary().get(datetime(l1))
    figsize = (7, 7)

    # plot longitude
    plt.figure(figsize=figsize)
    plt.imshow(l1.longitude)
    plt.title('Longitude')
    plt.colorbar()
    conftest.savefig(request)

    # Plot embedded MERIS
    plt.figure(figsize=figsize)
    plt.imshow(l1[variable])
    plt.title(f'{variable} ({get_unit(l1[variable])}) | MERIS')
    plt.colorbar()
    conftest.savefig(request)

    # Plot interpolated data (in time and lat/lon)
    interpolated = anc[variable].compute().interp(
        latitude=l1.latitude,
        longitude=l1.longitude,
        method=method,
    )
    plt.figure(figsize=figsize)
    plt.imshow(interpolated)
    plt.colorbar()
    plt.title(f'{variable} ({get_unit(anc[variable])}) | ERA5')
    conftest.savefig(request)
