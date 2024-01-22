#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import dask.array as da
import dask
import xarray as xr
import numpy as np
from tests.test_common import make_dataset
from eoread.process import Blockwise
from eoread.reader.olci import Level1_OLCI
from eoread.process import coerce_dtype, blockwise_method, blockwise_function
import eoread
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
    l1 = make_dataset()

    res = da.blockwise(
        f, [0, 1, 2,],
        l1.rho_toa.data, [3, 1, 2],
        l1.lat.data, [1, 2],
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
    l1 = make_dataset()

    proc = Processor()
    res = da.blockwise(
        proc.run, [1, 2, 3],
        l1.rho_toa.data, [1, 2, 3],
        )
    print(res.shape)
    print(res.compute().shape)
    assert res.compute().shape == res.shape
    np.testing.assert_allclose(res, l1.rho_toa**2)


@pytest.mark.parametrize('n', [50, None])
def test_blockwise_1(n):
    '''
    Simple case: one input, one output
    '''
    def f(sza):
        return sza

    l1 = make_dataset().isel(
        x=slice(0, n),
        y=slice(0, n))

    blk = Blockwise(
        f,
        dims_blockwise=('x', 'y'),
        dims_out=[('x', 'y')],
        dtypes=['float64'])
    res = blk(l1.lat)

    assert not isinstance(res.data, np.ndarray) # output should be a dask array
    assert res.shape == l1.lat.shape
    np.testing.assert_allclose(res, l1.lat)


@pytest.mark.parametrize('i', [0, 1, 2])
@pytest.mark.parametrize('size,chunksize', [(200, 100),
                                            (1, 1)])
def test_blockwise_2(i, size, chunksize):
    '''
    Multiple inputs, multiple outputs
    '''
    def f(Rtoa, sza):
        return Rtoa, sza, (sza > 0).astype('uint8')

    l1 = make_dataset((size, size), chunks=chunksize)
    dims2 = ('x', 'y')
    dims3 = ('bands', 'x', 'y')
    blk = Blockwise(
        f,
        dims_blockwise=('x', 'y'),
        dims_out=[dims3, dims2, dims2],
        dtypes=['float64', 'float64', 'uint8'])
    res1 = blk(l1.rho_toa, l1.lat)

    assert blk.dtype_coerce == np.dtype('float64')

    res2 = f(l1.rho_toa.compute(), l1.lat.compute())
    np.testing.assert_allclose(res1[i], res2[i])


def test_blockwise_3():
    '''
    Multiple inputs, no output
    '''
    def f(rho_toa, lat):
        assert rho_toa.ndims == 3
        assert lat.ndims == 2
        return None

    l1 = make_dataset()

    blk = Blockwise(
        f,
        dims_blockwise=('x', 'y'),
        dims_out=[],
        dtypes=[])

    blk(l1.rho_toa, l1.lat)


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


def test_decorator_1():
    dims2 = ('x', 'y')
    dims3 = ('bands', 'x', 'y')
    @blockwise_function(
        dims_blockwise=dims2,
        dims_out=[dims3, dims2, dims2],
        dtypes=['float32', 'float64', 'uint8'])
    def f(lat, rho_toa):
        return rho_toa.astype('float32'), lat, (lat > 0).astype('uint8')
    
    l1 = make_dataset()
    res = f(l1.lat, l1.rho_toa)

    np.testing.assert_allclose(res[0], l1.rho_toa.compute())
    np.testing.assert_allclose(res[1], l1.lat.compute())
    np.testing.assert_allclose(res[2], (l1.lat > 0).compute())


def test_with_a_processing_class():
    '''
    Use a class to share initializations, then process in a method
    '''
    class Process:
        def __init__(self):
            pass

        def run(self, rho_toa, lat):
            return rho_toa+lat, lat**2

    l1 = make_dataset()
    proc = Process()
    res0 = Blockwise(
        proc.run,
        dims_blockwise=('x', 'y'),
        dims_out=[('bands', 'x', 'y'),
                  ('x', 'y')],
        dtypes=['float64', 'float64'])(l1.rho_toa, l1.lat)
    res1 = proc.run(l1.rho_toa, l1.lat)

    np.testing.assert_allclose(res0[0], res1[0])
    np.testing.assert_allclose(res0[1], res1[1])


def test_with_a_processing_class_decorator_1():
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
            dims_blockwise=('x', 'y'),
            dims_out=[('bands', 'x', 'y'),
                      ('x', 'y')],
            dtypes=['float64', 'float64'])
        def run_blockwise(self, rho_toa, lat):
            return self.run(rho_toa, lat)

    l1 = make_dataset()
    proc = Process()
    res0 = proc.run_blockwise(l1.rho_toa, l1.lat)
    res1 = proc.run(l1.rho_toa, l1.lat)

    np.testing.assert_allclose(res0[0], res1[0])
    np.testing.assert_allclose(res0[1], res1[1])

    assert proc.nexec > 0


def test_with_a_processing_class_decorator_2():
    '''
    Use a class to share initializations, then process in a method

    Method returns nothing here
    '''
    class Process:
        def __init__(self):
            self.nexec = 0

        def run(self, Rtoa, sza):
            self.nexec += 1
            return Rtoa+sza, sza**2

        @blockwise_method(
            dims_blockwise=('x', 'y'),
            dims_out=[],
            dtypes=[])
        def run_blockwise(self, Rtoa, sza):
            return None

    l1 = make_dataset()
    proc = Process()
    proc.run_blockwise(l1.rho_toa, l1.lat)
    proc.run(l1.rho_toa, l1.lat)

    assert proc.nexec > 0


@pytest.mark.parametrize('kind', ['function', 'method'])
@pytest.mark.parametrize('use_dask', [False, True])
@pytest.mark.parametrize('expand_dims', [False, True])
def test_map_blocks(use_dask, kind, expand_dims):
    """
    Main test of eoread.map_blocks
    """
    def run(rho_toa, lat, flags):
        test1 = rho_toa+lat
        test2 = lat**2
        test3 = np.random.rand(3, *lat.shape)
        flags = flags + (lat > 0)
        return test1, test2, test3, flags

    class Processor:
        def __init__(self):
            self.a = 0
        def run(self, rho_toa, lat, flags):
            return run(rho_toa, lat, flags)
    outputs = (('test1', ('bands', 'x', 'y')),
               ('test2', ('x', 'y')),
               ('test3', ('z', 'x', 'y')),
               ('flags', ('x', 'y')))

    func = {'function': run,
            'method': Processor().run,
            }[kind]

    l1 = make_dataset()
    l1['flags'] = xr.zeros_like(l1.lat, dtype='int')
    if expand_dims:
        l1 = l1.sel(x=0, y=0).expand_dims(dim='x', axis=-1).expand_dims(dim='y', axis=-1)
    if not use_dask:
        l1 = l1.compute()

    # First method: keep input variables
    r1 = eoread.map_blocks(
        func,
        l1,
        outputs=outputs,
    )
    assert 'lat' not in r1
    assert 'test1' in r1
    assert 'test1' in l1
    print(r1)
    r1.compute()
    l1.compute()
    assert (r1.flags > 0).any()

    # Second method: pass arguments explicitly
    r2 = eoread.map_blocks(
        func,
        args={
            'rho_toa': l1.rho_toa,
            'lat': l1.lat,
            'flags': l1.flags,
        },
        outputs=outputs,
    )
    assert 'lat' not in r2
    assert 'test1' in r2
    r2.compute()
    assert (r2.flags > 0).any()

def test_map_blocks_single_output():
    """
    Test eoread.map_blocks with a single output
    """
    l1 = make_dataset()
    def process(rho_toa):
        return rho_toa*1.01
    eoread.map_blocks(
        process,
        l1,
        outputs=[('rho_toa_modif', l1.rho_toa.dims)],
    )


def test_apply_ufunc():
    '''
    xr.apply_ufunc can be used to map a numpy ufunc across blocks

    have to provide core dimensions (here, 'bands'), which are moved to the last dimension
    '''

    def run(rho_toa, lat):
        test1 = rho_toa+lat[:,:,None]
        test2 = lat**2
        return test1, test2, lat > 0

    l1 = make_dataset()
    l1['test1'], l1['test2'], l1['flags']= xr.apply_ufunc(
        run,
        l1.rho_toa, l1.lat,
        dask='parallelized',
        input_core_dims=[['bands'], []],
        output_core_dims=[['bands'], [], []],
        )
    l1.compute()

