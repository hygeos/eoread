#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from functools import wraps

import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


class AtIndex:
    '''
    An array-like using DataArray `idx` to index DataArray `A` along dimension `idx_name`

    Example:
        A: DataArray (nbands x detectors)
        idx: DataArray (rows x columns)
        Results in A[idx]: (nbands x rows x columns)
    '''
    def __init__(self, A, idx, idx_name):
        # dimensions to be indexed by this object
        self.dims = tuple(sum([[x] if not x == idx_name else list(idx.dims) for x in A.dims], []))
        # ... and their shape
        shape = tuple(sum([[A.shape[i]] if not x == idx_name else list(idx.shape)
                           for i, x in enumerate(A.dims)], []))
        self.shape = shape
        self.dtype = A.dtype
        self.A = A
        self.idx = idx
        self.pos_dims_idx = [i for i, x in enumerate(self.dims) if x in idx.dims]
        self.idx_name = idx_name

    def __getitem__(self, key):
        # first index idx using the appropriate dimensions in key
        idx = self.idx[tuple([key[i] for i in self.pos_dims_idx])].values

        # then index A using the remaining dimensions
        return self.A.values[tuple([key[i] if k != self.idx_name else idx
                                    for i, k in enumerate(self.A.dims)])]


class Interpolator:
    '''
    An array-like object to interpolate 2-dim array `A` to new `shape`

    Uses coordinates `tie_rows` and `tie_columns`.
    '''
    def __init__(self, shape, A, method='linear'):
        self.shape = shape
        self.dtype = A.dtype
        self.A = A
        assert A.ndim == 2
        self.ndim = A.ndim
        self.method = method
        self.dims = A.dims

    def __getitem__(self, key):
        ret = self.A.interp(
            {
                self.dims[0]: np.arange(self.shape[0])[key[0]],
                self.dims[1]: np.arange(self.shape[1])[key[1]],
            },
            method=self.method,
        )
        # dtype is not preserved by interp
        # coercing to self.dtype
        return ret.compute().astype(self.dtype)


class Repeat:
    '''
    Repeat elements of `A`

    Parameters:
    A: DataArray to repeat
    repeats: tuple of int (number of repeats along each dimension)
    '''
    def __init__(self, A, repeats):
        self.shape = tuple([s*r for (s, r) in zip(A.shape, repeats)])
        self.ndim = len(self.shape)
        self.repeats = repeats
        self.dtype = A.dtype
        self.A = A

    def __getitem__(self, key):
        indices = [np.arange(self.shape[i], dtype='int')[k]//self.repeats[i]
                   for i, k in enumerate(key)]
        X, Y = np.meshgrid(*indices)
        return np.array(self.A)[X, Y].transpose()


def DataArray_from_array(A, dims, chunks):
    '''
    Returns a DataArray (backed by dask) from array-like `A`

    Arguments:
    - A: array-like
    - dims: named dimensions of DataArray (ex: ('x', 'y'))
    - chunks: int
    '''
    assert not isinstance(chunks, dict)
    return xr.DataArray(
        da.from_array(
            A,
            meta=np.array([], A.dtype),
            chunks=chunks,
        ),
        dims=dims)

def rectBivariateSpline(A, shp):
    '''
    Bivariate spline interpolation of array A to shape shp.

    Fill NaNs with closest values, otherwise RectBivariateSpline gives no
    result.
    '''
    xin = np.arange(shp[0], dtype='float32') / (shp[0]-1) * A.shape[0]
    yin = np.arange(shp[1], dtype='float32') / (shp[1]-1) * A.shape[1]

    x = np.arange(A.shape[0], dtype='float32')
    y = np.arange(A.shape[1], dtype='float32')

    invalid = np.isnan(A)
    if invalid.any():
        # fill nans
        # see http://stackoverflow.com/questions/3662361/
        ind = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        A = A[tuple(ind)]

    f = RectBivariateSpline(x, y, A)

    return f(xin, yin).astype('float32')


def len_slice(s, l):
    '''
    returns the length of slice `s` applied to an iterable of lenght `l`

    (thus, `len(range(l)[s])`)
    '''
    # https://stackoverflow.com/questions/36188429
    start, stop, step = s.indices(l)

    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def convert_for_nc(value):
    """
    Convert value to a number, a string, an ndarray or a list/tuple of numbers/strings
    for serialization to netCDF files
    """
    if isinstance(value, bytes):
        return value.decode()
    else:
        return value


def timeit(func):
    """
    A decorator to print the execution time of a callable
    """
    @wraps(func)
    def timed(*args, **kwargs):
        start = datetime.now()
        try:
            return func(*args, **kwargs)
        finally:
            t = datetime.now() - start
            print(f"Total execution time of {func}: {t}")
    return timed


def floor_dt(dt, delta):
    """
    Round `dt` to the previous time period `delta`

    Args:
    -----
    dt: datetime

    delta: timedelta
    """
    # https://stackoverflow.com/questions/13071384/python-ceil-a-datetime-to-next-quarter-of-an-hour
    return dt - (dt - datetime.min) % delta


def ceil_dt(dt, delta):
    """
    Round `dt` to the next time period `delta`

    Args:
    -----
    dt: datetime

    delta: timedelta
    """
    return dt + (datetime.min - dt) % delta
