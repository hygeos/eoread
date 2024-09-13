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

from ..common import DataArray_from_array
from eoread.utils import tools


def clean_attrs(A):
    '''
    Remove all '\x00' from attribute values
    '''
    def clean(x):
        if isinstance(x, str):
            return x.rstrip('\x00')
        else:
            return x
    return {k: clean(v) for k, v in A.items()}


class HDF4_ArrayLike:
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

    def __getitem__(self, keys):
        return self.sds.__getitem__(keys)

def load_hdf4(filename, trim_dims=False, chunks=1000, lazy=False):
    """
    Loads a hdf4 file as a lazy xarray object
    """
    hdf = SD(str(filename))
    ds = xr.Dataset()
    for name, (dims, shp, dtype, index) in hdf.datasets().items():
        sds = hdf.select(name)
        if lazy:
            data = HDF4_ArrayLike(sds)
        else:
            data = HDF4_ArrayLike(sds)[:]
        ds[name] = DataArray_from_array(
            data,
            dims,
            chunks=chunks,
        )
        ds[name].attrs.update(clean_attrs(sds.attributes()))

    ds.attrs.update(clean_attrs(hdf.attributes()))

    if trim_dims:
        return tools.trim_dims(ds)
    else:
        return ds


