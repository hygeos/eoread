#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from matplotlib import pyplot as plt
from eoread.legacy.goesng import Level1_GOESNG, config
from eoread.hdf4 import load_hdf4
from . import conftest
from . import generic

product_l1 = '/archive'
product_l2 = '/mnt/ceph/data/LAN/'


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture()
def product_goesng(chunks):
    return Level1_GOESNG(product_l1, chunks=chunks)


def test_load_hdf4():
    ds = load_hdf4(config['auxfile'])
    ds.Latitude[:15,:100:3].compute()
    ds.Latitude[-10:,:-10:3].compute()



@pytest.mark.parametrize('chunksize', [1000, {'x': 1000, 'y': -1}])
def test_instantiate(chunksize):
    ds = Level1_GOESNG(local_config.goes_sample_file,
                       chunksize=chunksize,
                       cloudmask=True)
    print(ds)


@pytest.mark.parametrize('var', [
    'VIS_004',
    'VIS_006',
    'VIS_008',
    'latitude',
    'longitude',
    'vza',
    'vaa',
    'sza',
    'saa',
])
def test_preview(request, var):
    l1 = Level1_GOESNG(local_config.goes_sample_file)
    print(l1)
    l1 = l1.isel(
        y=slice(None, None, 100),
        x=slice(None, None, 100),
        )

    plt.imshow(l1[var])
    plt.colorbar()
    conftest.savefig(request)


def test_clouds(request):
    l1 = Level1_GOESNG(local_config.goes_sample_file, cloudmask=True)
    print(l1)
    l1 = l1.isel(
        y=slice(4000, 4900),
        x=slice(5000, 6100),
        )

    plt.imshow(l1['VIS_004'])
    plt.colorbar()
    conftest.savefig(request)

    plt.imshow(l1.flags)
    plt.colorbar()
    conftest.savefig(request)


def test_main():
    ds = Level1_GOESNG(local_config.goes_sample_file)
    eo.init_Rtoa(ds)
    generic.test_main(ds)

def test_read(param, indices, scheduler):
    ds = Level1_GOESNG(local_config.goes_sample_file)
    eo.init_Rtoa(ds)
    generic.test_read(ds, param, indices, scheduler)

def test_subset():
    ds = Level1_GOESNG(local_config.goes_sample_file)
    generic.test_subset(ds)