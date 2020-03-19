#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime
import pytest
from eoread import eo
from eoread.era5 import ERA5
from eoread.olci import Level1_OLCI
from tests.products import products as p, get_path
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


def test_get():
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
    Fetch and interpolate ancillary data for an OLCI product
    """
    l1 = Level1_OLCI(get_path(p['prod_S3_L1_20190430'])).sel(
        rows=slice(None, None, 10),
        columns=slice(None, None, 10))
    anc = ERA5().get(eo.datetime(l1))
    print(anc)

    anc.u10.compute().interp(
        time=eo.datetime(l1)
    ).interp(
        latitude=l1.latitude,
        longitude=l1.longitude,
        method=method,
    ).plot()
    conftest.savefig(request)
