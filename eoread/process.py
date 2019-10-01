#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Blockwise process wrapper using dask array's `blockwise` function
'''

from functools import wraps
import dask.array as da
import numpy as np
import xarray as xr


class Blockwise:
    '''
    Call a function by blocks using da.blockwise

    ufunc: the universal function to apply
    dims_blockwise (tuple)
        Blockwise dimensions, like ('height', 'width')
        These dimensions should be the last ones of all arrays, input and output
    dims_out (list of tuples)
        Output dimensions of ufunc, like [('band', 'height', 'width'), ('height', 'width')]
    dtypes (list of dtypes)
        dtypes of the output, like ['float32', 'uint16']

    Note1: the ufunc should align the dimensions to the last dimension
    Note2: the block dimensions should be the last ones.
        They are determined based on the input arrays.

    Example:
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
        # res[0], res[1] and res[2] are the lazy outputs of f
    '''
    def __init__(self, ufunc, dims_blockwise, dims_out, dtypes):
        self.ufunc = ufunc
        self.dims_blockwise = dims_blockwise
        self.dims_out = dims_out
        self.dtypes = dtypes

        assert len(dims_out) == len(dtypes)
        for i, dims in enumerate(dims_out):
            assert dims[-len(dims_blockwise):] == dims_blockwise, \
                f'The last dimensions of all output arrays (output ' \
                f'#{i} has dimensions {dims}) should be the ' \
                f'blockwise ones {dims_blockwise}.' 
        # coerce all outputs to the largest dtype
        self.dtype_coerce = sorted(self.dtypes, key=lambda x: np.dtype(x).itemsize)[-1]


    def run(self, *args):
        '''
        The callable to be run in parallel

        Wraps self.ufunc and merges all outputs in a single dask array cast to uint8
        '''
        args = list(args)
        for i, a in enumerate(args):
            # for each disappearing dimension, all chunks along
            # that dimension are passed as a list
            # => assuming that all disappearing dimensions are not chunked
            if isinstance(a, list):
                assert len(a) == 1
                args[i] = a[0]

        # apply the function
        res = self.ufunc(*args)

        if len(self.dtypes) == 1:
            assert not isinstance(res, tuple), f'Expected single output, received {len(res)}'
            res = [res]
        
        assert len(res) == len(self.dtypes), \
            f'Expected {len(self.dtypes)} outputs, received {len(res)}.'

        # stack all results
        res_stacked = []
        for i, r in enumerate(res):
            assert r.dtype == self.dtypes[i], \
                f'output {i}/{len(res)-1}: expected dtype {self.dtypes[i]} but received {r.dtype}'

            new_shp = r.shape[-len(self.dims_blockwise):]
            new_shp = (r.size // np.prod(new_shp),) + new_shp
            res_stacked.append(coerce_dtype(r, self.dtype_coerce).reshape(new_shp))

        res_stacked = da.concatenate(res_stacked)

        return res_stacked

    def __call__(self, *args):
        '''
        Apply the run function in parallel using apply_gufunc and split the results
        '''
        blockwise_args = []
        ndimblk = len(self.dims_blockwise)

        # all dimensions in the input DataArrays
        dims_shape_input = dict(set(sum([list(zip(a.dims, a.shape)) for a in args], [])))
        dims_input = list(dims_shape_input.keys())

        for a in args:
            assert isinstance(a, xr.DataArray)

            # The last dimensions of the input arrays should be the blockwise ones
            assert a.dims[-ndimblk:] == self.dims_blockwise, \
                f'Expected blockwise dimensions to be {self.dims_blockwise}, ' \
                f'but found {a.dims[-ndimblk:]}'

            # Check that the chunked dimensions are the last ones
            chunked = [len(c) > 1 for c in a.chunks]
            assert False not in chunked[-ndimblk:],\
                f'Encountered a non-chunked dimension in following dimensions: {a.dims[-ndimblk:]}'
            assert True not in chunked[:-ndimblk], \
                f'Encountered a chunked dimension in following dimensions: {a.dims[:-ndimblk]}'

            blockwise_args.append(a.data)
            blockwise_args.append([dims_input.index(d)+1 for d in a.dims])

        # calculate the size of the stacked dimension
        sizes_stacked = []
        for i, dims in enumerate(self.dims_out):
            s = np.prod(
                [dims_shape_input[d] for d in dims[:-ndimblk]],
                dtype='int')

            sizes_stacked.append(s)

        # process using da.blockwise
        res = da.blockwise(
            self.run,
            [0] + [dims_input.index(d)+1     # the raveled dimension
                   for d in self.dims_blockwise],  # takes dimension number 0
            *blockwise_args,
            new_axes={0: sum(sizes_stacked)},
            meta=np.array([], dtype=self.dtype_coerce),  # otherwise f is called
                                                         # immediately with dummy parameters
            )

        # Unstack the results
        unstacked = []
        pos = 0   # current index along the stacked dimension
        for i, dims in enumerate(self.dims_out):
            s = sizes_stacked[i]
            shp = [dims_shape_input[d] for d in self.dims_out[i]]
            data = coerce_dtype(res[slice(pos, pos+s), ...], self.dtypes[i])

            unstacked.append(
                xr.DataArray(data.reshape(shp), dims=self.dims_out[i])
                )

            pos += s

        if len(self.dtypes) == 1:
            return unstacked[0]
        else:
            return tuple(unstacked)


def coerce_dtype(A, dtype):
    '''
    coerce A to another dtype while retaining A.shape:
    - if dtype is larger than A.dtype, pad with zeros
    - if dtype has same size as A.dtype, just return a view
      with new dtype
    - if dtype is smaller than A.dtype, assume each memory
      element is padded with zeros and return a view of the
      array with new dtype and each element unpadded

    Coercing to a larger dtype, and then to the original dtype,
    returns the original array
    '''
    dtype = np.dtype(dtype)

    if dtype.itemsize > A.dtype.itemsize:
        # coercing to a larger dtype: pad with zeros
        n = dtype.itemsize // A.dtype.itemsize
        assert dtype.itemsize == n * A.dtype.itemsize

        B = np.zeros(A.shape + (n,), dtype=A.dtype)
        B[..., 0] = A[...]
        return B.view(dtype)[..., 0]

    elif dtype.itemsize < A.dtype.itemsize:
        # coercing to a smaller dtype: assume it is padded with zeros
        # extract the memory elements corresponding to dtype
        n = A.dtype.itemsize // dtype.itemsize
        assert A.dtype.itemsize == n * dtype.itemsize
        return A.view(dtype).reshape(A.shape + (n,))[..., 0]

    else:
        return A.view(dtype)


def blockwise_function(dims_blockwise, dims_out, dtypes):
    '''
    Decorates a function for blockwise processing

    dims_blockwise (tuple)
        Blockwise dimensions, like ('height', 'width')
        These dimensions should be the last ones of all arrays, input and output
    dims_out (list of tuples)
        Output dimensions of ufunc, like [('band', 'height', 'width'), ('height', 'width')]
    dtypes (list of dtypes)
        dtypes of the output, like ['float32', 'uint16']
    '''
    def decorator(f):
        @wraps(f)
        def wrapper(*args):
            blk = Blockwise(f, dims_blockwise, dims_out, dtypes).__call__
            return blk(*args)
        return wrapper
    return decorator


def blockwise_method(dims_blockwise, dims_out, dtypes):
    '''
    Decorates a method for blockwise processing

    dims_blockwise (tuple)
        Blockwise dimensions, like ('height', 'width')
        These dimensions should be the last ones of all arrays, input and output
    dims_out (list of tuples)
        Output dimensions of ufunc, like [('band', 'height', 'width'), ('height', 'width')]
    dtypes (list of dtypes)
        dtypes of the output, like ['float32', 'uint16']
    '''
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args):
            def run(*a):
                return method(self, *a)
            return Blockwise(run, dims_blockwise, dims_out, dtypes)(*args)
        return wrapper
    return decorator