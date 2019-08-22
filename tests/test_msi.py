#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from eoread.msi import Level1_MSI
from tests import products as p
from tests.products import sentinel_product, sample_data_path
from eoread.common import GeoDatasetAccessor



@pytest.mark.parametrize('product,resolution',
                         [(p.prod_S2_L1_20190419,res) for res in ['20', '60']])   # FIXME: 10m leads to memory error
def test_msi(sentinel_product, resolution):
    ds = Level1_MSI(sentinel_product, resolution)
    print(ds)