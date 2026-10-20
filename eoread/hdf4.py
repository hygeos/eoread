#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read HDF4 files as dask arrays

An alternative is to use open_dataset with pynio engine,
but it is complex to install.
"""

import operator
from threading import Lock

import xarray as xr
import numpy as np

from pyhdf.SD import SD, SDC
from dask import base
from dask.array import Array, array
from dask.highlevelgraph import HighLevelGraph

from core import tools


def clean_attrs(A):
    """
    Remove all null terminators from HDF4 attribute string values.
    
    HDF4 files often contain null-terminated strings in attributes.
    This function strips trailing '\x00' characters from string values.
    
    Parameters
    ----------
    A : dict
        Dictionary of HDF4 attributes.
        
    Returns
    -------
    dict
        Dictionary with cleaned attribute values (non-strings unchanged).
    """
    def clean(x):
        if isinstance(x, str):
            return x.rstrip('\x00')
        else:
            return x
    return {k: clean(v) for k, v in A.items()}


class _HDF4File:
    """Own one pyhdf ``SD`` handle and the lock shared by all its datasets.

    pyhdf's C API is not thread-safe, so every block read goes through the same
    lock.  The handle is closed when the last array referencing it is garbage
    collected (dask tasks keep the wrapper, which keeps this object alive).
    """

    def __init__(self, hdf):
        self.hdf = hdf
        self.lock = Lock()

    def close(self) -> None:
        hdf, self.hdf = getattr(self, 'hdf', None), None
        if hdf is not None:
            try:
                hdf.close()
            except Exception:
                pass

    def __del__(self):
        self.close()


class HDF4_ArrayLike:
    """
    Array-like wrapper for HDF4 Scientific Dataset (SDS) objects.
    
    Provides a NumPy-compatible interface to HDF4 datasets with proper
    dtype, shape, and ndim attributes. Enables lazy loading via Dask.
    
    Attributes:
        sds: HDF4 Scientific Dataset object
        dtype: NumPy dtype of the dataset
        shape: Tuple describing array dimensions
        ndim: Number of dimensions
    """
    def __init__(self, sds, handle: _HDF4File):
        self.sds = sds
        self.handle = handle
        self.dtype = {
            SDC.FLOAT32: np.dtype('float32'),
            SDC.FLOAT64: np.dtype('float64'),
            SDC.INT8: np.dtype('int8'),
            SDC.UINT8: np.dtype('uint8'),
            SDC.INT16: np.dtype('int16'),
            SDC.UINT16: np.dtype('uint16'),
            SDC.INT32: np.dtype('int32'),
            SDC.UINT32: np.dtype('uint32'),
        }[sds.info()[3]]
        shp = sds.info()[2]
        if hasattr(shp, '__len__'):
            self.shape = tuple(shp)
        else:
            self.shape = (shp,)
        self.ndim = len(self.shape)

    def __getitem__(self, keys):
        """
        Read data from HDF4 dataset using NumPy-style indexing.
        
        Parameters
        ----------
        keys : slice or tuple of slices
            Slice or index specification.
            
        Returns
        -------
        np.ndarray
            NumPy array with requested data.
        """
        with self.handle.lock:
            return self.sds.__getitem__(keys)


def _chunks_for(shape: tuple[int, ...], chunks) -> tuple[tuple[int, ...], ...]:
    """Normalize a chunk size spec to dask's per-dimension chunk tuples."""
    if chunks is None:
        chunks = 500
    if isinstance(chunks, (int, np.integer)):
        chunks = (int(chunks),) * len(shape)
    if len(chunks) != len(shape):
        raise ValueError(
            f'chunks {chunks!r} do not match the array shape {shape}')
    out = []
    for size, chunk in zip(shape, chunks):
        chunk = int(chunk)
        if chunk == -1:  # dask convention: one chunk for this dimension
            chunk = int(size)
        chunk = max(1, min(chunk, int(size)))
        out.append(tuple(min(chunk, size - i) for i in range(0, size, chunk)))
    return tuple(out)


def _lazy_array(wrapper: 'HDF4_ArrayLike', filename: str, name: str,
                chunks=None) -> Array:
    """
    Wrap an ``HDF4_ArrayLike`` in a lazy, windowed dask array.

    Passing the wrapper to ``dask.array.array`` (i.e. any ``np.asarray`` path)
    makes numpy iterate over it, which poisons the underlying pyhdf session:
    every subsequent read on that file then fails, including selecting another
    dataset.  The block graph is therefore built by hand -- one task per
    rectangular block, each performing a plain slice read at compute time.
    This is what keeps the pyhdf fallback genuinely lazy: the previous
    single-block graph read the full SDS before any rechunking.
    """
    shape = tuple(int(s) for s in wrapper.shape)
    out_chunks = _chunks_for(shape, chunks)
    key_name = 'hdf4-' + base.tokenize(filename, name, shape, out_chunks)

    dsk = {}
    for key, slices in _block_keys(key_name, out_chunks):
        dsk[key] = (operator.getitem, wrapper, slices)

    graph = HighLevelGraph.from_collections(key_name, dsk)
    return Array(graph, key_name, chunks=out_chunks,
                 dtype=wrapper.dtype, shape=shape)


def _block_keys(name: str, chunks: tuple[tuple[int, ...], ...]):
    """Yield (task key, slice tuple) for every block of the chunk grid."""
    starts = []
    for dim_chunks in chunks:
        dim_starts, offset = [], 0
        for size in dim_chunks:
            dim_starts.append(offset)
            offset += size
        starts.append(dim_starts)

    for index in np.ndindex(*(len(s) for s in starts)):
        slices = tuple(slice(starts[axis][i], starts[axis][i] + chunks[axis][i])
                       for axis, i in enumerate(index))
        yield (name, *index), slices


def load_hdf4(filename, trim_dims=False, lazy=False, chunks=None):
    """
    Load an HDF4 file as an xarray Dataset with optional lazy loading.
    
    This function provides an alternative to xarray's pynio engine, which is
    complex to install. All datasets are wrapped in Dask arrays for consistency.
    
    Parameters
    ----------
    filename : Path or str
        Path to the HDF4 file (.hdf).
    trim_dims : bool, optional
        If True, removes unused dimensions from the dataset. Default is False.
    lazy : bool, optional
        If True, uses lazy loading (Dask arrays). If False, loads data immediately. Default is False.
    chunks : int or tuple of int, optional
        Block size per dimension for the lazy arrays (default: 500). Only used
        when ``lazy=True``. Each dask task reads one block from the HDF4 file,
        so this controls both the read window size and the output chunking.
        
    Returns
    -------
    xr.Dataset
        Dataset with variables from the HDF4 file. Each variable includes its
        original HDF4 attributes, and the dataset has global file attributes.
        
    Examples
    --------
    >>> ds = load_hdf4('MODIS_L1B.hdf', lazy=True)
    >>> ds = load_hdf4('ancillary_data.hdf', trim_dims=True)
    """
    hdf = SD(str(filename))
    # One handle per file: it owns the SD session, the shared pyhdf lock and
    # the lifetime of the session (closed when the last lazy array is freed).
    handle = _HDF4File(hdf)
    ds = xr.Dataset()
    for name, val in hdf.datasets().items():
        sds = hdf.select(name)
        dims = val[0]
        data = HDF4_ArrayLike(sds, handle)
        attrs = clean_attrs(sds.attributes())
        if lazy:
            arr = _lazy_array(data, str(filename), name, chunks=chunks)
        else:
            arr = array(data[:])
        ds[name] = xr.DataArray(arr, dims=dims)
        ds[name].attrs.update(attrs)

    ds.attrs.update(clean_attrs(hdf.attributes()))

    if trim_dims:
        return tools.trim_dims(ds)
    else:
        return ds


