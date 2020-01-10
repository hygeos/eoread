#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Generic tests implementation
"""

import os
import tempfile

import dask
import numpy as np
import pytest

from eoread import eo
from eoread.naming import naming as n


@pytest.fixture(params=[
    n.Rtoa,
    n.lat,
    n.lon,
    n.vza,
    n.sza,
    n.raa,
    n.flags,
])
def param(request):
    return request.param

@pytest.fixture(params=[
    (slice(120, 130), slice(122, 135)),
    (slice(100, 130, 3), slice(122, 135, 2)),
    (20, slice(120, 130)),
    (20, 20),
])
def indices(request):
    return request.param


def test_main(ds):

    # check dimensions
    assert n.rows in ds.dims
    assert n.columns in ds.dims
    assert ds[n.Rtoa].dims == n.dim3
    assert n.wav in ds

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


def test_read(ds, param, indices):
    idx1, idx2 = indices
    assert param in ds

    with dask.config.set(scheduler='single-threaded'):
        # v = da.compute()
        expected_dtype = np.dtype(n.expected_dtypes[param])

        res = ds[param].sel(rows=idx1, columns=idx2).compute()
        assert ds[param].dtype == expected_dtype,\
            f'Dtype error: expected {expected_dtype}, found {ds[param].dtype}'
        assert res.dtype == expected_dtype,\
            f'Dtype error: expected {expected_dtype}, found {res.dtype} (after compute)'


def test_subset(ds):
    sub = ds.isel(
        rows=slice(300, 400),
        columns=slice(500, 570))

    with tempfile.TemporaryDirectory() as tmpdir,\
            dask.config.set(scheduler='single-threaded'):
        target = os.path.join(tmpdir, 'test.nc')
        sub.to_netcdf(target)
