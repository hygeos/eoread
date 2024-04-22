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

from eoread.utils.tools import datetime
from eoread.utils.naming import naming as n


@pytest.fixture(params=[
    'single-threaded',
    'threads',
    # 'processes',
])
def scheduler(request):
    return request.param

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
    (20, 20),                                   # two ints
    (slice(120, 130), slice(122, 135)),         # two slices
    (20, slice(120, 130)),                      # int and slice
    (slice(510, 530, 3), slice(-25, -20, 3)),   # two slices with steps
])
def indices(request):
    return request.param


def test_main(ds, radiometry='reflectance', angle_data=False):
    assert radiometry in ['radiance','reflectance'], \
    f"radiometry arg should be set to radiance or reflectance, not {radiometry}"

    # test chunks consistency
    ds.chunks

    # check dimensions
    assert n.rows in ds.dims
    assert n.columns in ds.dims
    if radiometry: 
        if n.Rtoa in ds: assert ds[n.Rtoa].dims == n.dim3 
        if n.BT in ds: assert ds[n.BT].dims == n.dim3_tir
        if n.Rtoa not in ds and n.BT not in ds: 
            raise ValueError(f'{n.Rtoa} or {n.BT} is missing')
    else: 
        if n.Ltoa_tir in ds: assert ds[n.Ltoa_tir].dims == n.dim3_tir
        elif n.Ltoa in ds  : assert ds[n.Ltoa].dims == n.dim3
        if n.Ltoa_tir not in ds and n.Ltoa not in ds:
            raise ValueError(f'{n.Ltoa} or {n.Ltoa_tir} is missing')

    # # spectral data
    # # either just provide wav (per-band central wavelength)
    # # or per-pixel wavelength + central wavelength
    # assert n.wav in ds
    # if ds.wav.ndim == 3:
    #     assert n.cwav in ds
    #     assert ds.cwav.ndim == 1
    # else:
    #     assert (ds.wav.ndim == 1)


    # check that attributes exist and are not empty
    assert ds.datetime
    datetime(ds)
    assert ds.platform
    assert ds.sensor
    assert ds.product_name
    assert ds.resolution
    assert ds.input_directory

    # test datasets
    assert n.flags in ds
    assert ds[n.flags].dtype == n.flags_dtype

    # TODO: test footprint
    assert n.lat in ds
    assert n.lon in ds
    
    # test angle data
    if angle_data:
        assert n.vaa in ds
        assert n.vza in ds
        assert n.saa in ds
        assert n.sza in ds


def test_read(ds, param, indices, scheduler):
    idx1, idx2 = indices
    assert param in ds

    with dask.config.set(scheduler=scheduler):
        # v = da.compute()
        expected_dtype = np.dtype(n.expected_dtypes[param])

        res = ds[param].sel({n.rows:idx1, n.columns:idx2}).compute()
        assert ds[param].dtype == expected_dtype,\
            f'Dtype error: expected {expected_dtype}, found {ds[param].dtype}'
        assert res.dtype == expected_dtype,\
            f'Dtype error: expected {expected_dtype}, found {res.dtype} (after compute)'
        
        # for the "stepped" indices, check that result is consistent with "non-stepped"
        # (also with an offset)
        if (isinstance(idx1, slice) and isinstance(idx2, slice) and idx1.step and idx2.step):
            A = ds[param].sel({n.rows:idx1, n.columns:idx2}).compute()
            B = ds[param].sel({
                    n.rows:slice(idx1.start-1, idx1.stop),
                    n.columns:slice(idx2.start-1, idx2.stop),
                }).compute()[..., 1::idx1.step, 1::idx2.step]
            np.testing.assert_allclose(A, B)


def test_subset(ds):
    sub = ds.isel({
        n.rows:slice(300, 400),
        n.columns:slice(500, 570)})

    with tempfile.TemporaryDirectory() as tmpdir,\
            dask.config.set(scheduler='single-threaded'):
        target = os.path.join(tmpdir, 'test.nc')
        sub.to_netcdf(target)
