#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from eoread.msi import Level1_MSI
from tests import products as p
from tests.products import sentinel_product, sample_data_path
from eoread.common import GeoDatasetAccessor



@pytest.mark.parametrize('product,resolution',
                         [(p.prod_S2_L1_20190419, res) for res in ['10', '20', '60']])
def test_msi_merged(sentinel_product, resolution):
    l1 = Level1_MSI(sentinel_product, resolution)
    print(l1)
    assert 'Rtoa_443' not in l1
    assert 'Rtoa' in l1

    # Try to access latitude
    sub = l1.sel(rows=slice(50, 100), columns=slice(100, 150))
    sub.latitude.compute()


@pytest.mark.parametrize('product,resolution',
                         [(p.prod_S2_L1_20190419, res) for res in ['10', '20', '60']])
def test_msi_split(sentinel_product, resolution):
    l1 = Level1_MSI(sentinel_product, resolution, split=True)
    print(l1)
    assert 'Rtoa_443' in l1
    assert 'Rtoa' not in l1
    print(l1.Rtoa_490)
