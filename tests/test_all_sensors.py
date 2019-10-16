#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import tempfile
import pytest
import dask
from eoread.msi import Level1_MSI
from eoread.olci import Level1_OLCI, get_l2_flags, get_valid_l2_pixels
from eoread.naming import Naming
from eoread import eo
from tests import products as p
from tests.products import sentinel_product, sample_data_path
dask.config.set(scheduler='single-threaded')


@pytest.mark.parametrize('idx1, idx2', [
    (slice(20, 30), slice(20, 30)),
    (20, slice(20, 30)),
    (20, 20),
    ])
@pytest.mark.parametrize('param', ['Rtoa', 'latitude', 'longitude', 'vza', 'sza'])   # FIXME: raa
@pytest.mark.parametrize('Reader,product,kwargs', [
    (Level1_OLCI, p.prod_S3_L1_20190430, {'init_reflectance': True}),
    (Level1_MSI, p.prod_S2_L1_20190419, {}),
    ])
def test(sentinel_product, Reader, kwargs, idx1, idx2, param):
    '''
    Various verifications to check the consistency of all products
    '''
    ds = Reader(sentinel_product, **kwargs)
    n = Naming()

    # presence of datasets
    assert n.Rtoa in ds

    # test reading
    # (slice of pixel)
    if param == 'Rtoa':
        ds[param][0, idx1, idx2].compute()
    else:
        ds[param][idx1, idx2].compute()

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
    eo.datetime(ds)
    assert n.platform in ds.attrs
    assert n.sensor in ds.attrs
    assert n.product_name in ds.attrs

    # subset
    sub = ds.isel(rows=slice(1000, 1100), columns=slice(500, 570))

    # check that product can be written
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'test.nc')
        sub.to_netcdf(target)
    
    # TODO: test footprint?
