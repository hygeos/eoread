#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from eoread.reader.venus import Level1_VENUS, Level2_VENUS, get_SRF, get_sample
from eoread.utils.graphics import plot_srf
from . import generic
from eoread import eo
from . import conftest
from matplotlib import pyplot as plt

import pytest
import xarray as xr


product_l1 = get_sample('level1')
product_l2 = get_sample('level2')


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def VENUS_product(chunks):
    return Level1_VENUS(product_l1, chunks=chunks)

@pytest.mark.parametrize('split', [True, False])
def test_instantiation(split, chunks):
    Level1_VENUS(product_l1, split=split, chunks=chunks)


@pytest.mark.parametrize('param', ['sza', 'vza', 'saa', 'vaa', 'latitude', 'longitude'])
def test_msi_merged(VENUS_product, param):
    l1 = VENUS_product
    print(l1)
    assert 'Rtoa_420' not in l1
    assert 'Rtoa' in l1

    # check parameter consistency through windowing
    xr.testing.assert_allclose(
        l1[param][1000, 500],
        l1.isel(y=slice(1000, None),
                x=slice(500, None))[param][0, 0])

    xr.testing.assert_allclose(
        l1[param][1000:1010, 500:510],
        l1.isel(y=slice(1000, None),
                x=slice(500, None))[param][:10, :10])


@pytest.mark.parametrize('band', ['Rtoa_420', 'Rtoa_490', 'Rtoa_865'])
def test_msi_split(band):
    l1 = Level1_VENUS(product_l1, split=True)
    print(l1)
    assert 'Rtoa_420' in l1
    assert 'Rtoa' not in l1

    assert l1[band][:10, :10].values.shape == (10, 10)

    xr.testing.assert_allclose(
            l1[band][:600, :600].compute()[500:550, 450:550],
            l1.sel(y=slice(500, 550), x=slice(450, 550))[band],
            )
    

def test_main():
    l1 = Level1_VENUS(product_l1, chunks=500)
    generic.test_main(l1)


@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(VENUS_product, scheduler):
    eo.init_geometry(VENUS_product)
    generic.test_read(VENUS_product, 'Rtoa', (1000,1000), scheduler)


def test_subset(VENUS_product):
    generic.test_subset(VENUS_product)

from dask import config
def test_plot(request):
    ds = Level1_VENUS(product_l1)
    eo.init_geometry(ds)

    for desc, data in [
        ("rho_toa865", ds.Rtoa.sel(bands=865)),
        ('latitude', ds.latitude),
        ('longitude', ds.longitude),
        ('flags', ds.flags),
        ('sza', ds.sza),
        ('vza', ds.vza),
        ('raa', ds.raa),
    ]:
        plt.figure()
        plt.title(desc)
        with config.set(scheduler='sync'):
            data.thin(x=10, y=10).plot()
            conftest.savefig(request)


def test_srf(request):
    srf = get_SRF()
    plot_srf(srf)
    conftest.savefig(request, bbox_inches="tight")

def test_level2(chunks):
    Level2_VENUS(product_l2, chunks=chunks)