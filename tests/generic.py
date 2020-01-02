#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Generic tests implementation
"""

import os
import tempfile

import pytest

from eoread import eo
from eoread.naming import naming as n


@pytest.fixture(params=[
    'Rtoa',
    'latitude',
    'longitude',
    'vza',
    'sza',
    'raa',
])
def param(request):
    return request.param

@pytest.fixture(params=[
    (slice(20, 30), slice(20, 30)),
    (20, slice(20, 30)),
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

    # test reading
    # (slice of pixel)
    if ds[param].ndim == 3:
        ds[param][0, idx1, idx2].compute()
    elif ds[param].ndim == 2:
        ds[param][idx1, idx2].compute()
    else:
        raise Exception(f'Unexpected number of dimensions ({param}: {ds[param].ndim})')


def test_subset(ds):
    sub = ds.isel(
        rows=slice(300, 400),
        columns=slice(500, 570))

    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'test.nc')
        sub.to_netcdf(target)
