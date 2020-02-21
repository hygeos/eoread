#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
import xarray as xr
import numpy as np
import dask.array as da
from eoread.common import AtIndex, Repeat, len_slice
from eoread import eo


def make_dataset():
    l1 = xr.Dataset()
    bands = [412, 443, 490, 510, 560]
    shp2 = (10, 12)
    shp3 = (len(bands), 10, 12)
    dims2 = ('x', 'y')
    dims3 = ('bands', 'x', 'y')
    l1['rho_toa'] = xr.DataArray(np.random.randn(*shp3), dims=dims3)
    l1['rho_w'] = xr.DataArray(np.random.randn(*shp3), dims=dims3)
    l1['lat'] = xr.DataArray(np.random.randn(*shp2), dims=dims2)
    l1['lon'] = xr.DataArray(np.random.randn(*shp2), dims=dims2)
    l1 = l1.assign_coords(bands=bands)

    # set some attributes
    l1.attrs['sensor'] = 'OLCI'
    l1.rho_w.attrs['unit'] = 'dimensionless'

    return l1


def test_merge():
    # TODO: deprecate merge
    # create a dataset
    l1 = xr.Dataset()
    bands = [412, 443, 490, 510, 560]
    bnames = [f'Rtoa_{b}' for b in bands]
    for b in bnames:
        l1[b] = xr.DataArray(np.eye(10), dims=('x', 'y'))
    print(l1)

    l1m = eo.merge(l1, bnames, 'Rtoa', 'bands', coords=bands)
    print(l1m)


def test_split():
    l1 = make_dataset()
    print(l1)
    l1s = eo.split(l1, 'bands')
    assert 'rho_toa_412' in l1s
    assert 'rho_w_412' in l1s
    assert 'rho_toa' not in l1s
    assert 'sensor' in l1s.attrs
    assert 'unit' in l1s.rho_w_412.attrs
    assert 'bands' in l1s.coords   # split should preserve coords
    print(l1s)


def test_split_dataarray():
    l1 = make_dataset()
    l1s = eo.split(l1.rho_w, 'bands')
    print(l1s)


def test_split_without_coords():
    l1 = make_dataset()

    with pytest.raises(AssertionError):
        eo.split(l1.rho_w, 'unknown_dim')

    l1 = l1.drop('bands')
    with pytest.raises(AssertionError):
        eo.split(l1.rho_w, 'bands')


def test_merge2():
    # create a dataset
    l1 = make_dataset()
    l1s = eo.split(l1, 'bands')
    l1m = eo.merge2(l1s)
    print('Original:', l1)
    print('Merged:', l1m)
    assert l1m.equals(l1)


def test_merge2_inconsistent_dimension():
    l1 = make_dataset()
    l1s = eo.split(l1, 'bands')
    eo.merge2(l1s, 'bands')
    l1s = l1s.rename({'rho_w_412': 'rho_w_413'})
    with pytest.raises(AssertionError):
        eo.merge2(l1s, 'bands')


@pytest.mark.parametrize('A', [
            np.eye(5),
            da.eye(5, chunks=2),
            np.random.rand(5, 10),
            ])
def test_repeat(A):
    rep = (2, 2)
    B = Repeat(A, rep)
    np.testing.assert_allclose(
        B[0, 0],
        A[0, 0])
    for i in range(rep[0]):
        for j in range(rep[1]):
            np.testing.assert_allclose(
                B[i::rep[0], j::rep[1]],
                A
                )

def test_da_from_array_meta():
    """
    Check that da.from_array has an argument `meta` (use a recent version of dask)

    Note: this argument avoids calling A.__getitem__ to retrieve its dtype
    """
    A = np.eye(10)
    da.from_array(
        A,
        chunks=(2, 2),
        meta=np.array([], A.dtype),
    )


def test_AtIndex_1():
    shp = (25, 25)
    A = xr.DataArray(
        np.random.random(100),
        dims=('index'),
    )
    idx = xr.DataArray(
        np.random.randint(0, 100, shp),
        dims=('x', 'y'),
    )

    AA = AtIndex(
        A,
        idx,
        'index'
    )

    assert AA.dims == ('x', 'y')
    assert AA.shape == shp
    np.testing.assert_allclose(AA[:, :], A[idx])


def test_AtIndex_2():
    shp = (25, 25)
    A = xr.DataArray(
        np.random.random((10, 100)),
        dims=('bands', 'index'),
    )
    idx = xr.DataArray(
        np.random.randint(0, 100, shp),
        dims=('x', 'y'),
    )

    AA = AtIndex(
        A,
        idx,
        'index'
    )

    assert AA.dims == ('bands', 'x', 'y')
    assert AA.shape == (10,) + shp
    np.testing.assert_allclose(AA[:, :, :], A[:, idx])


@pytest.mark.parametrize('l', [1, 10, 100])
@pytest.mark.parametrize('s', [slice(5), slice(5, 50), slice(5, None, 3), slice(5, 50, 7)])
def test_len_slice(s, l):
    assert len_slice(s, l) == len(range(l)[s])


@pytest.mark.parametrize('dimsA,shpA', [
    (('bands',), (10,)),
])
@pytest.mark.parametrize('dimsB,shpB', [
    (('bands', 'x', 'y'), (10, 4, 5)),
    (('x', 'bands', 'y'), (4, 10, 5)),
    (('x', 'y', 'bands'), (4, 5, 10)),
])
def test_broadcast(dimsA, shpA, dimsB, shpB):
    """
    test the function eo.broadcast
    """
    A = xr.DataArray(
        da.from_array(
            np.random.random(shpA),
        ),
        dims=dimsA,
    )
    B = xr.DataArray(
        da.from_array(
            np.random.random(shpB),
        ),
        dims=dimsB,
    )
    AA = eo.broadcast(A, B)
    assert np.isclose(
        AA.sel(bands=3).min(),
        AA.sel(bands=3).max(),
    )

    # test case where broadcasting has no effect
    BB = eo.broadcast(B, B)
    np.testing.assert_allclose(BB, B)


def test_raiseflag():
    flags = xr.DataArray(
        np.zeros((15, 15), dtype='uint16')
    )
    A = xr.DataArray(np.random.randn(15, 15))
    eo.raiseflag(flags, 'FLAG_1', 2, A > 0)
    with pytest.raises(AssertionError):
        eo.raiseflag(flags, 'FLAG_2', 2, A > 0.1)

    eo.raiseflag(flags, 'FLAG_2', 4, A > 0.1)
