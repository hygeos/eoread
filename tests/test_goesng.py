#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from matplotlib import pyplot as plt
from eoread.goesng import Level1_GOESNG, config
from eoread.hdf4 import load_hdf4
from . import conftest


config['auxfile'] = '/rfs/proj/SARGASSES/GOES/ANCILLARY/GOESNG-0750.1km.hdf'

def test_load_hdf4():
    ds = load_hdf4(config['auxfile'])
    ds.Latitude[:15,:100:3].compute()
    ds.Latitude[-10:,:-10:3].compute()

@pytest.mark.parametrize('var', ['VIS_004', 'VIS_006', 'VIS_008'])
def test_goesng(request, var):
    l1 = Level1_GOESNG('/rfs/proj/SARGASSES/GOES/L1_B.v1.01/2018_08_10/Emultic1kmNC4_goes16_201808102145.nc', chunksize=1000)
    print(l1)
    l1 = l1.isel(
        rows=slice(None, None, 100),
        columns=slice(None, None, 100),
        )
    
    plt.imshow(l1[var])
    plt.colorbar()
    conftest.savefig(request)


