#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
import xarray as xr
from eoread.msi import Level1_MSI
from tests import products as p
from tests.products import sentinel_product, sample_data_path



@pytest.mark.parametrize('product,resolution,param',
                         [(p.prod_S2_L1_20190419, res, param)
                          for res in ['10', '20', '60']
                          for param in ['sza', 'vza', 'saa', 'vaa', 'latitude', 'longitude']])
def test_msi_merged(sentinel_product, resolution, param):
    l1 = Level1_MSI(sentinel_product, resolution)
    print(l1)
    assert 'Rtoa_443' not in l1
    assert 'Rtoa' in l1

    # Try to access latitude
    sub = l1.sel(rows=slice(50, 100), columns=slice(100, 150))
    sub.latitude.compute()

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
        xr.testing.assert_allclose(
            l1[param].compute()[1000:1010, 500:510],
            l1.isel(rows=slice(1000, None),
                    columns=slice(500, None))[param][:10, :10])


@pytest.mark.parametrize('product,resolution',
                         [(p.prod_S2_L1_20190419, res) for res in ['10', '20', '60']])
def test_msi_split(sentinel_product, resolution):
    l1 = Level1_MSI(sentinel_product, resolution, split=True)
    print(l1)
    assert 'Rtoa_443' in l1
    assert 'Rtoa' not in l1
    print(l1.Rtoa_490)
