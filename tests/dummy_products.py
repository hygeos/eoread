import xarray as xr
import dask
import dask.array as da
from dask.array import apply_gufunc
import numpy as np
from time import sleep
import inspect



class ArrayLike:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = np.float32
        self.ndim = len(shape)

    def __getitem__(self, key):
        res = np.random.randn(*[len(range(self.shape[i])[k])
                                for i, k in enumerate(key)]).astype(self.dtype)
        return res

def dummy_level1():
    l1 = xr.Dataset()
    l1['Rtoa'] = (
        ('band', 'width', 'height'),
        dask.array.from_array(
            np.random.randn(5, 200, 200).astype('float32'),
            chunks=(-1, 100, 100)))
    l1['sza'] = (
        ('width', 'height'),
        dask.array.from_array(
            np.random.randn(200, 200).astype('float64'),
            chunks=(100, 100)))
    l1['sensor'] = 'OLCI'

    return l1
