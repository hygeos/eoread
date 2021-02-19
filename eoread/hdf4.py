#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read HDF4 files as dask arrays

An alternative is to use open_dataset with pynio engine,
but it is complex to install.
"""

import xarray as xr
from pyhdf.SD import SD, SDC
import numpy as np
from eoread.common import DataArray_from_array


class HDF4_ArrayLike:
    def __init__(self, sds):
        self.sds = sds
        self.dtype = {
            SDC.FLOAT32: np.dtype('float32'),
            SDC.FLOAT64: np.dtype('float64'),
            SDC.INT16: np.dtype('int16'),
            SDC.INT32: np.dtype('int32'),
        }[sds.info()[3]]
        self.shape = tuple(sds.info()[2])

    def __getitem__(self, keys):
        return self.sds.__getitem__(keys)

def load_hdf4(filename, chunks=1000):
    """
    Loads a hdf4 file as a lazy xarray object
    """
    hdf = SD(str(filename))
    ds = xr.Dataset()
    for name, (dims, shp, dtype, index) in hdf.datasets().items():
        sds = hdf.select(name)
        ds[name] = DataArray_from_array(
            HDF4_ArrayLike(sds),
            dims,
            chunks=chunks,
        )
        ds[name].attrs.update(sds.attributes())

    ds.attrs.update(hdf.attributes())

    return ds


