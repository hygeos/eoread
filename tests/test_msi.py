#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
import xarray as xr
from eoread.msi import Level1_MSI
from tests.products import products as p
from .generic import indices, param
from . import generic
from eoread import eo

resolutions = ['10', '20', '60']


@pytest.fixture
def product():
    return p['prod_S2_L1_20190419']

@pytest.fixture(params=resolutions)
def resolution(request):
    return request.param

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def S2_product(product, resolution, chunks):
    return Level1_MSI(product['path'], resolution, chunks=chunks)


@pytest.mark.parametrize('split', [True, False])
def test_instantiation(product, resolution, split, chunks):
    Level1_MSI(product['path'], resolution, split=split, chunks=chunks)


@pytest.mark.parametrize('param', ['sza', 'vza', 'saa', 'vaa', 'latitude', 'longitude'])
def test_msi_merged(S2_product, param):
    l1 = S2_product
    print(l1)
    assert 'Rtoa_443' not in l1
    assert 'Rtoa' in l1

    # check parameter consistency through windowing
    xr.testing.assert_allclose(
        l1[param][1000, 500],
        l1.isel(rows=slice(1000, None),
                columns=slice(500, None))[param][0, 0])

    xr.testing.assert_allclose(
        l1[param][1000:1010, 500:510],
        l1.isel(rows=slice(1000, None),
                columns=slice(500, None))[param][:10, :10])

    if resolution == '60':
        full = l1[param].compute(scheduler='single-threaded')
        xr.testing.assert_allclose(
            full[1000:1010, 500:510],
            l1.isel(rows=slice(1000, None),
                    columns=slice(500, None))[param][:10, :10])


@pytest.mark.parametrize('band', ['Rtoa_443', 'Rtoa_490', 'Rtoa_865'])
def test_msi_split(product, band, resolution):
    l1 = Level1_MSI(product['path'], resolution, split=True)
    print(l1)
    assert 'Rtoa_443' in l1
    assert 'Rtoa' not in l1

    assert l1[band][:10, :10].values.shape == (10, 10)

    xr.testing.assert_allclose(
            l1[band][:600, :600].compute()[500:550, 450:550],
            l1.sel(rows=slice(500, 550), columns=slice(450, 550))[band],
            )



def test_main(S2_product):
    generic.test_main(S2_product)


@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(S2_product, param, indices, scheduler):
    eo.init_geometry(S2_product)
    generic.test_read(S2_product, param, indices, scheduler)


def test_subset(S2_product):
    generic.test_subset(S2_product)
