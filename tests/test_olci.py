#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from eoread.olci import Level1_OLCI, Level2_OLCI, olci_init_spectral
from eoread.msi import Level1_MSI
from tests import products as p
from tests.products import sentinel_product, sample_data_path
from eoread.common import GeoDatasetAccessor
from eoread.olci import get_l2_flags, get_valid_l2_pixels


@pytest.mark.parametrize('product', [p.prod_S3_L1_20190430])
def test_olci_level1(sentinel_product):
    ds = Level1_OLCI(sentinel_product)
    olci_init_spectral(ds)
    ds.eo.init(['Rtoa', 'geometry'])
    print(ds)

    # test method contains
    lat = ds.latitude[100, 100]
    lon = ds.longitude[100, 100]
    assert ds.eo.contains(lat, lon)
    assert not ds.eo.contains(lat, lon+180)

@pytest.mark.parametrize('product', [p.prod_S3_L1_20190430])
def test_split_merge(sentinel_product):
    ds = Level1_OLCI(sentinel_product)
    print(ds)
    ds = ds.eo.sub_rect(55, 56, 18, 19)
    split = ds.eo.split('Ltoa', out_vars = ['Ltoa_'+str(band) for band in ds.coords['bands'].data], split_axis = 'bands')
    print(split)
    merge = split.eo.merge([var for var in split.variables if 'Ltoa_' in var], 'Ltoa', 'bands')
    print(merge)


@pytest.mark.parametrize('product', [p.prod_S3_L2_20190612])
def test_olci_level2(sentinel_product):
    l2 = Level2_OLCI(sentinel_product)
    print(l2)


@pytest.mark.parametrize('product', [p.prod_S3_L2_20190612])
def test_olci_level2_flags(sentinel_product):
    l2 = Level2_OLCI(sentinel_product)

    get_l2_flags(l2.wqsf)
    get_valid_l2_pixels(l2.wqsf)