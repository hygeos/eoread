#!/usr/bin/env python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

import dask.array as da
import numpy as np
import xarray as xr

from warnings import warn
warn('The module `common` will be deprecated from eoread package. Please use functions from `core` instead.', DeprecationWarning)


class Interpolator:
    '''
    An array-like object to interpolate a 2-D array `A` to a new `shape`.

    Uses xarray's ``.interp()`` under the hood, called lazily via
    ``__getitem__`` so that dask can consume it chunk-by-chunk.
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
        return ret.values.astype(self.dtype)


def DataArray_from_array(A, dims, chunks):
    '''
    Returns a dask-backed DataArray from an array-like ``A``.

    Arguments
    ---------
    A : array-like
        Must have ``.shape`` and ``.dtype`` attributes (e.g. an ``Interpolator``).
    dims : tuple of str
        Dimension names, e.g. ``('rows', 'columns')``.
    chunks : int
        Chunk size along each dimension.
    '''
    assert not isinstance(chunks, dict)
    return xr.DataArray(
        da.from_array(
            A,
            meta=np.array([], A.dtype),
            chunks=chunks,
        ),
        dims=dims,
    )


def len_slice(s, l):
    """
    Return the length of slice `s` applied to an iterable of length `l`.

    (thus, `len(range(l)[s])`)
    """
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


@contextmanager
def timeit(desc=None, verbose=True):
    """
    A decorator/context to print the execution time of a callable

    Examples
    --------
    1) As a decorator:
        @timeit()
        def f():
            ...
    2) As a context manager:
        with timeit() as ti:
            sleep(1)
        print(ti())
    """
    start = perf_counter()
    try:
        yield lambda: perf_counter() - start
    finally:
        elapsed = perf_counter() - start
        if verbose:
            desc_msg = '' if desc is None else f' ({desc})'
            msg = f"Execution time{desc_msg}: {elapsed:.4f}s"
            print(msg)


def floor_dt(dt, delta):
    """
    Round `dt` to the previous time period `delta`

    Parameters
    ----------
    dt : datetime
    delta : timedelta
    """
    # https://stackoverflow.com/questions/13071384/python-ceil-a-datetime-to-next-quarter-of-an-hour
    return dt - (dt - datetime.min) % delta


def ceil_dt(dt, delta):
    """
    Round `dt` to the next time period `delta`

    Parameters
    ----------
    dt : datetime
    delta : timedelta
    """
    return dt + (datetime.min - dt) % delta


def bin_centers(N, vmin=0, vmax=0):
    """
    Returns the center of N bins equally spaced in [vmin, vmax]
    """
    return np.linspace(vmin, vmax, 2*N+1)[1::2]