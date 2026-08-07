#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read HDF4 files as dask arrays

An alternative is to use open_dataset with pynio engine,
but it is complex to install.
"""

import xarray as xr
import numpy as np

from pyhdf.SD import SD, SDC
from dask.array import array

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
    def __init__(self, sds):
        self.sds = sds
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
        return self.sds.__getitem__(keys)

def load_hdf4(filename, trim_dims=False, lazy=False):
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
    ds = xr.Dataset()
    for name, val in hdf.datasets().items():
        sds = hdf.select(name)
        dims = val[0]
        data = HDF4_ArrayLike(sds)
        if not lazy: data = data[:]
        ds[name] = xr.DataArray(array(data), dims=dims)
        ds[name].attrs.update(clean_attrs(sds.attributes()))

    ds.attrs.update(clean_attrs(hdf.attributes()))

    if trim_dims:
        return tools.trim_dims(ds)
    else:
        return ds


