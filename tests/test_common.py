#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
import xarray as xr
import numpy as np
import dask.array as da
from eoread.common import Repeat


def test_merge():
    # create a dataset
    l1 = xr.Dataset()
    bands = [412, 443, 490, 510, 560]
    bnames = [f'Rtoa_{b}' for b in bands]
    for b in bnames:
        l1[b] = xr.DataArray(np.eye(10), dims=('x', 'y'))
    print(l1)

    l1m = l1.eo.merge(bnames, 'Rtoa', 'bands', coords=bands)
    print(l1m)


def test_split():
    l1 = xr.Dataset()
    l1['Rtoa'] = xr.DataArray(np.zeros((5, 10, 10)),
                              dims=('bands', 'x', 'y'),
                              coords={'bands': [412, 443, 490, 510, 560]}
                              )
    print(l1)
    l1s = l1.eo.split('Rtoa')
    print(l1s)


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
