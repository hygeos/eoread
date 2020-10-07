#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Blockwise process wrapper using dask array's `blockwise` function
'''

from functools import wraps
from inspect import signature
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
        for x, dims in enumerate(dims_out):
            assert dims[-len(dims_blockwise):] == dims_blockwise, \
                f'The last dimensions of all output arrays (output ' \
                f'#{x+1}/{len(dims_out)} has dimensions {dims}) should be the ' \
                f'blockwise ones {dims_blockwise}.'
        # coerce all outputs to the largest dtype
        if self.dtypes:
            self.dtype_coerce = sorted(self.dtypes, key=lambda x: np.dtype(x).itemsize)[-1]
        else:   # empty self.dtypes
            self.dtype_coerce = None


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
            if hasattr(r, 'dtype'):
                rdtype = r.dtype
            else: # in case of memoryview
                rdtype = r.base.dtype

            assert rdtype == self.dtypes[i], \
                f'output {i+1}/{len(res)}: expected dtype {self.dtypes[i]} ' + \
                f'but received {rdtype} (in blockwise call to {self.ufunc})'

            new_shp = r.shape[-len(self.dims_blockwise):]
            new_shp = (r.size // np.prod(new_shp),) + new_shp
            res_stacked.append(coerce_dtype(r, self.dtype_coerce).reshape(new_shp))

        res_stacked = da.concatenate(res_stacked)

        return res_stacked

    def __call__(self, *args):
        '''
        Apply the run function in parallel using `da.blockwise` and split the results
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
                f'but found {a.dims[-ndimblk:]} (in {a})'

            # Check that the chunked dimensions are the last ones
            assert a.chunks is not None, f'Error in blockwise call: {a.name} is not chunked'
            chunked = [len(c) > 1 for c in a.chunks]
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
            name=f'blockwise_{self.ufunc}',
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

    if hasattr(A, 'dtype'):
        Adtype = A.dtype
    else: # in case of memoryview
        Adtype = A.base.dtype

    if dtype.itemsize > Adtype.itemsize:
        # coercing to a larger dtype: pad with zeros
        n = dtype.itemsize // Adtype.itemsize
        assert dtype.itemsize == n * Adtype.itemsize

        B = np.zeros(A.shape + (n,), dtype=Adtype)
        B[..., 0] = A[...]
        return B.view(dtype)[..., 0]

    elif dtype.itemsize < Adtype.itemsize:
        # coercing to a smaller dtype: assume it is padded with zeros
        # extract the memory elements corresponding to dtype
        n = Adtype.itemsize // dtype.itemsize
        assert Adtype.itemsize == n * dtype.itemsize
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
            @wraps(method)
            def run(*a):
                return method(self, *a)
            return Blockwise(run, dims_blockwise, dims_out, dtypes)(*args)
        return wrapper
    return decorator


def map_blocks(
        func,
        ds=None,
        args=None,
        outputs=None,
    ):
    """
    Apply a ufunc `func` by blocks to dataset `ds`.

    Arguments:
    ----------
    func: user provided universal function

    ds: xr.Dataset or None.
        Dataset containing the input variables, and which is also updated with the results.
        if None, use an empty Dataset (in which case, the input variables are
        expected to be provided as args).
        The variable names to be provided to `func` are inferred from the function signature.
        To provide other variables, pass them as args.
        To avoid modifying the input variables, pass a copy of ds.

    outputs: iterable of tuples (name, dims)
        where `dims` is the dimensions of the output variable `name`
        Example:
            [('C', ('lambda', 'x', 'y')),  # `func` creates new variables C and D
             ('D', ('x', 'y'))]

    args: dict or None
        dictionary containing the explicit variables.

    About dask
    ----------

    if any of the input variables is backed by dask, `func` is applied block
    by block, using xr.map_blocks
    otherwise (if no input variable uses dask array), `func` is applied directly.

    Returns:
    -------

    A xr.Dataset containing only the results (they are also included in `ds`)

    Example:
    --------

    `ds` is the input dataset containing the variables `A` and `B`.
    Define the function:
    >>> def process(A, B):
    ...     [...]
    ...     return C, D

    Apply it to `ds`, using variables from defined in the function signature (`A` and `B`).
    Stores the results (`C` and `D`) back in ds.
    >>> map_blocks(process, ds,
    ...     outputs=[
    ...         ('C', ('lambda', 'x', 'y')),
    ...         ('D', ('x', 'y')),
    ...         ])

    Apply process by passing explicitly the relevant DataArrays `ds.A` and `ds.B`.
    By not passing ds, only the variables `C` and `D` are provided in results.
    >>> result = map_blocks(
    ...     process,
    ...     args={
    ...         'A': ds.A,
    ...         'B': ds.B
    ...     },
    ...     outputs=[
    ...         ('C', ds.A.dims),  # output variable 'C' has same dimensions as 'A'
    ...         ('D', ds.B.dims),  # output variable 'D' has same dimensions as 'B'
    ...         ])
    """
    # get the input variable names from the function signature
    sig = signature(func)
    list_inputs = list(sig.parameters)

    # initialize input dataset ds_in
    if ds is None:
        ds_in = xr.Dataset()
    else:
        assert isinstance(ds, xr.Dataset)
        ds_in = ds.copy()  # do not alter the input dataset

    # Add other variables to ds_in from keywords
    if args is not None:
        for k, v in args.items():
            assert k in list_inputs
            ds_in[k] = v

    # check that all input variables are in ds_in
    for x in list_inputs:
        assert x in ds_in, f'{x} is missing'

    # determine the output dimensions
    out_dims = []
    for k, d in outputs:
        assert isinstance(d, tuple)
        out_dims.append(d)

    @wraps(func)
    def wrapper(block):
        """
        Apply func to the relevant variables of the dataset, and store the
        resulting variables back in the dataset
        """
        args = (block[x].data for x in list_inputs)
        res = func(*args)

        if not isinstance(res, tuple):
            res = (res,)
        assert len(res) == len(outputs), \
            f'function has {len(res)} outputs, but {len(outputs)} were expected.'

        res_ds = xr.Dataset()
        for i, r in enumerate(res):
            varname = outputs[i][0]
            res_ds[varname] = (out_dims[i], r)

        return res_ds

    if True in [isinstance(ds_in[x].data, da.Array) for x in ds_in]:
        # if any of the input DataArrays is a dask array, use xr.map_blocks
        ret = xr.map_blocks(wrapper, ds_in)
    else:
        # none of the input DataArray is a dask Array: run the wrapper directly
        ret = wrapper(ds_in)
    
    if ds is not None:
        for (k, _) in outputs:
            ds[k] = ret[k]
    
    return ret
