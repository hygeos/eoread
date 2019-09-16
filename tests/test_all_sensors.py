#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from eoread.olci import Level1_OLCI
from eoread.msi import Level1_MSI
from tests import products as p
from tests.products import sentinel_product, sample_data_path
from eoread.common import GeoDatasetAccessor
from eoread.olci import get_l2_flags, get_valid_l2_pixels
from eoread.naming import Naming



@pytest.mark.parametrize('Reader,product', [
    (Level1_OLCI, p.prod_S3_L1_20190430),
    (Level1_MSI, p.prod_S2_L1_20190419),
    ])
def test(sentinel_product, Reader):
    '''
    Various verifications to check the consistency of all products
    '''
    ds = Reader(sentinel_product)
    n = Naming()

    if not n.Rtoa in ds:
        ds = ds.eo.init_Rtoa()

    # datasets
    assert n.Rtoa in ds
    # TODO: test reading slices, and pixels

    # check dimensions
    assert n.rows in ds.dims
    assert n.columns in ds.dims
    assert ds[n.Rtoa].dims == n.dim3

    # check chunking
    if n.Ltoa in ds:
        assert len(ds[n.Ltoa].chunks[0]) == 1, 'Ltoa is chunked along dimension `bands`'
    assert len(ds[n.Rtoa].chunks[0]) == 1, 'Rtoa is chunked along dimension `bands`'

    # check attributes
    assert n.datetime in ds.attrs
    assert n.platform in ds.attrs
    assert n.sensor in ds.attrs
