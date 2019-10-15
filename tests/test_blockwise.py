#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from tests.dummy_products import dummy_level1
import dask.array as da
import dask
import xarray as xr
import numpy as np
from eoread.process import Blockwise
from eoread.olci import Level1_OLCI
from eoread.process import coerce_dtype, blockwise_method, blockwise_function
from tests import products as p
from tests.products import sentinel_product, sample_data_path
dask.config.set(scheduler='single-threaded')


@pytest.mark.parametrize('A,dtype', [
    (np.random.randint(100, size=(3, 4, 5), dtype='uint8'), 'float64'),
    (np.random.randint(100, size=(3, 4, 5), dtype='uint8'), 'float32'),
    (np.random.randint(100, size=(3, 4, 5), dtype='uint8'), 'uint16'),
    (np.random.randint(100, size=(3, 4, 5), dtype='uint8'), 'uint8'),
    (np.random.rand(3, 4, 5).astype('float32'), 'float64'),
    (np.random.rand(3, 4, 5).astype('float32'), 'float32'),
])
def test_coerce_dtype(A, dtype):
    b = coerce_dtype(A, dtype)
    a1 = coerce_dtype(b, A.dtype)

    assert A.shape == a1.shape
    np.testing.assert_allclose(A, a1)


def test_blockwise_basic():
    '''
    test using dask array blockwise function
    
    Note: dimensions in blockwise can also be specified
    with tuples of  ints, unrelated to the dims sizes.
    '''
    def f(*args):
        Rtoa_chunks = args[0]
        sza = args[1]
        Rtoa = Rtoa_chunks[0]    # all chunks along dimension `i`
                                 # are passed as a list because
                                 # this dimensions disappears from the output.
        print('Applying f to', Rtoa.shape, sza.shape)
        return (Rtoa*1.01+sza)[:2, :, :]
    l1 = dummy_level1()

    res = da.blockwise(
        f, [0, 1, 2,],
        l1.Rtoa.data, [3, 1, 2],
        l1.sza.data, [1, 2],
        # dtype='float32',
        new_axes={0: 2},
        meta=np.array([], dtype='float32'), # otherwise f is called
                                            # immediately with dummy parameters
        )
    print(res.shape)
    print(res.compute().shape)
    assert res.compute().shape == res.shape


def test_blockwise_class():
    '''
    test applying dask array blockwise to a class method
    '''
    class Processor:
        def __init__(self):
            self.nrun = 0
        def run(self, Rtoa):
            print('Apply', Rtoa.shape)
            self.nrun += 1
            return Rtoa**2
    l1 = dummy_level1()

    proc = Processor()
    res = da.blockwise(
        proc.run, [1, 2, 3],
        l1.Rtoa.data, [1, 2, 3],
        )
    print(res.shape)
    print(res.compute().shape)
    assert res.compute().shape == res.shape
    np.testing.assert_allclose(res, l1.Rtoa**2)


@pytest.mark.parametrize('n', [50, None])
def test_blockwise_1(n):
    '''
    Simple case: one input, one output
    '''
    def f(sza):
        return sza

    l1 = dummy_level1().isel(
        width=slice(0, n),
        height=slice(0, n))

    blk = Blockwise(
        f,
        dims_blockwise=('width', 'height'),
        dims_out=[('width', 'height')],
        dtypes=['float64'])
    res = blk(l1.sza)

    assert res.shape == l1.sza.shape
    np.testing.assert_allclose(res, l1.sza)


@pytest.mark.parametrize('i', [0, 1, 2])
def test_blockwise_2(i):
    '''
    Multiple inputs, multiple outputs
    '''
    def f(Rtoa, sza):
        return Rtoa, sza, (sza > 0).astype('uint8')

    l1 = dummy_level1()
    dims2 = ('width', 'height')
    dims3 = ('band', 'width', 'height')
    blk = Blockwise(
        f,
        dims_blockwise=('width', 'height'),
        dims_out=[dims3, dims2, dims2],
        dtypes=['float32', 'float64', 'uint8'])
    res1 = blk(l1.Rtoa, l1.sza)

    assert blk.dtype_coerce == np.dtype('float64')

    res2 = f(l1.Rtoa.compute(), l1.sza.compute())
    np.testing.assert_allclose(res1[i], res2[i])


@pytest.mark.parametrize('i', [0, 1, 2])
def test_showcase(i):
    '''
    The same, but with a minimal reproducible example.
    '''
    def f(x, y):
        return x, y, (y > 0).astype('uint8')

    dims2 = ('dim1_block', 'dim2_block')
    dims3 = ('dim0', 'dim1_block', 'dim2_block')
    ds = xr.Dataset()
    ds['x'] = (
        ('dim0', 'dim1_block', 'dim2_block'),
        dask.array.from_array(
            np.random.randn(5, 200, 200).astype('float32'),
            chunks=(-1, 100, 100)))
    ds['y'] = (
        ('dim1_block', 'dim2_block'),
        dask.array.from_array(
            np.random.randn(200, 200).astype('float64'),
            chunks=(100, 100)))
    res = Blockwise(
        f,
        dims_blockwise=dims2,
        dims_out=[dims3, dims2, dims2],
        dtypes=['float32', 'float64', 'uint8']
    )(ds.x, ds.y)

    # check output
    res2 = f(ds.x.compute(), ds.y.compute())
    np.testing.assert_allclose(res[i], res2[i])


def test_decorator():
    dims2 = ('width', 'height')
    dims3 = ('band', 'width', 'height')
    @blockwise_function(
        dims_blockwise=dims2,
        dims_out=[dims3, dims2, dims2],
        dtypes=['float32', 'float64', 'uint8'])
    def f(sza, Rtoa):
        return Rtoa, sza, (sza > 0).astype('uint8')
    
    l1 = dummy_level1()
    res = f(l1.sza, l1.Rtoa)

    np.testing.assert_allclose(res[0], l1.Rtoa.compute())
    np.testing.assert_allclose(res[1], l1.sza.compute())
    np.testing.assert_allclose(res[2], (l1.sza > 0).compute())


def test_with_a_processing_class():
    '''
    Use a class to share initializations, then process in a method
    '''
    class Process:
        def __init__(self):
            pass

        def run(self, Rtoa, sza):
            return Rtoa+sza, sza**2

    l1 = dummy_level1()
    proc = Process()
    res0 = Blockwise(
        proc.run,
        dims_blockwise=('width', 'height'),
        dims_out=[('band', 'width', 'height'),
                  ('width', 'height')],
        dtypes=['float64', 'float64'])(l1.Rtoa, l1.sza)
    res1 = proc.run(l1.Rtoa, l1.sza)

    np.testing.assert_allclose(res0[0], res1[0])
    np.testing.assert_allclose(res0[1], res1[1])


def test_with_a_processing_class_decorator():
    '''
    Use a class to share initializations, then process in a method
    '''
    class Process:
        def __init__(self):
            self.nexec = 0

        def run(self, Rtoa, sza):
            self.nexec += 1
            return Rtoa+sza, sza**2

        @blockwise_method(
            dims_blockwise=('width', 'height'),
            dims_out=[('band', 'width', 'height'),
                      ('width', 'height')],
            dtypes=['float64', 'float64'])
        def run_blockwise(self, Rtoa, sza):
            return self.run(Rtoa, sza)

    l1 = dummy_level1()
    proc = Process()
    res0 = proc.run_blockwise(l1.Rtoa, l1.sza)
    res1 = proc.run(l1.Rtoa, l1.sza)

    np.testing.assert_allclose(res0[0], res1[0])
    np.testing.assert_allclose(res0[1], res1[1])

    assert proc.nexec > 0
