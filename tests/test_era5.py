#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pytest
from eoread import eo
from eoread.era5 import ERA5
from eoread.meris import Level1_MERIS
from tests.products import products as p
from . import conftest


def test_download(request):
    """
    Download a single file
    """
    dt = datetime(2010, 1, 1, 1, 0, 0)
    anc = ERA5().download(dt)
    assert len(anc.time) == 1
    print(anc)
    anc.sp.isel(time=0).plot()
    conftest.savefig(request)


def test_get_single():
    """
    Download and concat multiple files
    """
    anc = ERA5().get(datetime(2010, 1, 1, 1, 0, 0))
    print(anc)


def test_get_tuple():
    """
    Download and concat multiple files
    """
    anc = ERA5().get((datetime(2010, 1, 1, 1, 0, 0),
                      datetime(2010, 1, 1, 2, 0, 0)))
    assert len(anc.time) == 2
    print(anc)


@pytest.mark.parametrize('method', ['nearest', 'linear'])
def test_interp(request, method):
    """
    Fetch and interpolate ancillary data for a MERIS product

    We test with an orbit that crosses the date line
    """
    l1 = Level1_MERIS(p['prod_meris_L1_20080701']['path']).sel(
        rows=slice(2000, 2600),
        )

    anc = ERA5().get(eo.datetime(l1))
    ws = np.sqrt(anc.u10**2 + anc.v10**2)

    print(ws)
    figsize = (7, 7)

    # plot longitude
    plt.figure(figsize=figsize)
    plt.imshow(l1.longitude)
    plt.title('Longitude')
    plt.colorbar()
    conftest.savefig(request)

    # Plot embedded MERIS
    plt.figure(figsize=figsize)
    plt.imshow(l1.horizontal_wind)
    plt.title('Horizontal wind')
    plt.colorbar()
    conftest.savefig(request)

    # Plot interpolated data (in time and lat/lon)
    interpolated = ws.compute().interp(
        time=eo.datetime(l1)
    ).interp(
        latitude=l1.latitude,
        longitude=l1.longitude,
        method=method,
    )
    plt.figure(figsize=figsize)
    plt.imshow(interpolated)
    plt.colorbar()
    plt.title('Interpolated in ERA5')
    conftest.savefig(request)
