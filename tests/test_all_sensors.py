#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import tempfile
import pytest
import dask
from eoread.msi import Level1_MSI
from eoread.olci import Level1_OLCI, get_valid_l2_pixels
from eoread.naming import naming as n
from eoread import eo
from tests import products as p
from tests.products import sentinel_product, sample_data_path

dask.config.set(scheduler='single-threaded')

list_products = [
    (Level1_OLCI, p.prod_S3_L1_20190430),
    (Level1_MSI, p.prod_S2_L1_20190419),
]


@pytest.mark.parametrize('Reader,product', list_products)
def test_instantiate(sentinel_product, Reader):
    Reader(sentinel_product)


@pytest.mark.parametrize('Reader,product', list_products)
def test_misc(sentinel_product, Reader):
    ds = Reader(sentinel_product)

    ds = eo.init_Rtoa(ds)

    # check dimensions
    assert n.rows in ds.dims
    assert n.columns in ds.dims
    assert ds[n.Rtoa].dims == n.dim3
    assert n.wav in ds

    # check chunking
    if n.Ltoa in ds:
        assert len(ds[n.Ltoa].chunks[0]) == 1, 'Ltoa is chunked along dimension `bands`'
    assert len(ds[n.Rtoa].chunks[0]) == 1, 'Rtoa is chunked along dimension `bands`'

    # check attributes
    assert n.datetime in ds.attrs
    eo.datetime(ds)
    assert n.platform in ds.attrs
    assert n.sensor in ds.attrs
    assert n.product_name in ds.attrs

    # test datasets
    assert n.flags in ds
    assert ds[n.flags].dtype == n.flags_dtype

    # TODO: test footprint


@pytest.mark.parametrize('idx1, idx2', [
    (slice(20, 30), slice(20, 30)),
    (20, slice(20, 30)),
    (20, 20),
    ])
@pytest.mark.parametrize('param', ['Rtoa', 'latitude', 'longitude', 'vza', 'sza', 'raa'])
@pytest.mark.parametrize('Reader,product', list_products)
def test_read(sentinel_product, Reader, idx1, idx2, param):
    ds = Reader(sentinel_product)

    ds = eo.init_Rtoa(ds)

    assert param in ds

    # test reading
    # (slice of pixel)
    if param == 'Rtoa':
        ds[param][0, idx1, idx2].compute()
    else:
        ds[param][idx1, idx2].compute()


@pytest.mark.parametrize('Reader,product', list_products)
def test_subset(sentinel_product, Reader):
    ds = Reader(sentinel_product)

    # subset
    sub = ds.isel(rows=slice(1000, 1100), columns=slice(500, 570))

    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'test.nc')
        sub.to_netcdf(target)
