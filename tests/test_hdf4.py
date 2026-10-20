#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the lazy HDF4 block graph (no product data needed)."""

import numpy as np

from eoread.hdf4 import _chunks_for, _lazy_array


class _ArrayLike:
    """Minimal HDF4 array-like: records the slice it is asked to read."""

    shape = (3, 10, 12)
    dtype = np.dtype('uint16')
    ndim = len(shape)

    def __init__(self):
        self.slices = []

    def __getitem__(self, keys):
        self.slices.append(keys)
        return np.arange(np.prod(self.shape),
                         dtype=self.dtype).reshape(self.shape)[keys]


def test_lazy_array_has_one_task_per_block():
    wrapper = _ArrayLike()
    arr = _lazy_array(wrapper, 'file.hdf', 'SDS', chunks=(1, 5, 6))

    assert arr.chunks == ((1, 1, 1), (5, 5), (6, 6))
    assert arr.numblocks == (3, 2, 2)
    np.testing.assert_array_equal(arr.compute(), np.arange(360).reshape(3, 10, 12)[:])

    # Every task reads one rectangular window; no full-array read is issued.
    assert len(wrapper.slices) == arr.numblocks[0] * arr.numblocks[1] * arr.numblocks[2]
    for keys in wrapper.slices:
        assert all(isinstance(key, slice) for key in keys)
        assert keys != tuple(slice(None) for _ in wrapper.shape)


def test_chunks_for_negative_one_is_single_chunk():
    assert _chunks_for((3, 10), (1, -1)) == ((1, 1, 1), (10,))
