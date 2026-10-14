#!/usr/bin/env python3
"""
Unit tests for the lazy blockwise resampling in eoread.tools / eoread.transforms.

The pre-lazy implementations (eager compute + full-grid core.interpolate.interp)
are embedded verbatim below as bit-exact references: the lazy rewrite must
reproduce them exactly (values, dims, dtype), while reading no pixel data at
construction time and only local source windows at compute time.
"""

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from core.geo.naming import names

from eoread.tools import spatial_resample
from eoread.transforms.angles import tiepoint_angles


# ---------------------------------------------------------------------------
# References: verbatim copies of the pre-lazy implementations
# ---------------------------------------------------------------------------

def _sync_upsample_reference(array, output_shape, chunks, method='linear'):
    """Old over-sampling branch of spatial_resample (eager), kept verbatim."""
    from core.interpolate import interp, Linear, Nearest
    from dask.array import meshgrid, linspace

    array = array.squeeze()
    dims = array.dims
    array = array.compute(scheduler='sync')
    coords = array.coords
    method = Linear if method == 'linear' else Nearest
    xy = meshgrid(*[
        linspace(coords[d].values[0], coords[d].values[-1], output_shape[d])
        for d in dims
    ])
    params = {d: method(xr.DataArray(xy[i], dims=('d1', 'd0'))) for i, d in enumerate(dims)}
    resampled = interp(array, **params)
    resampled = resampled.rename(d0=dims[0], d1=dims[1])
    return resampled.chunk(chunks)


def _sync_angles_reference(ds, angles, tie_dims=('tie_rows', 'tie_columns'), chunks=None):
    """Old tiepoint_angles loop (eager), kept verbatim."""
    from core.interpolate import interp, Linear
    ny = ds.sizes[str(names.rows)]
    nx = ds.sizes[str(names.columns)]
    tie_r, tie_c = tie_dims
    y = xr.DataArray(np.linspace(0, float(ds[tie_r][-1].values), ny),
                     dims=(str(names.rows),)).chunk(-1)
    x = xr.DataArray(np.linspace(0, float(ds[tie_c][-1].values), nx),
                     dims=(str(names.columns),)).chunk(-1)
    out = {}
    for name in angles:
        tie = ds[f'{name}_tie']
        full = interp(tie.compute(scheduler='sync'), **{tie_r: Linear(y), tie_c: Linear(x)})
        arr = full.transpose(str(names.rows), str(names.columns)).astype('float32')
        out[str(name)] = arr.chunk(chunks) if chunks is not None else arr
    return ds.assign(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(ny, nx, dtype=np.float64, seed=0, nan_patch=True):
    """Random 2D field, optionally with NaN patches (exercises NaN*0 poisoning)."""
    rng = np.random.default_rng(seed)
    a = (rng.random((ny, nx)) * 100).astype(dtype)
    if nan_patch and np.issubdtype(a.dtype, np.floating):
        a[3:8, 5:9] = np.nan
        a[-4:, -6:-2] = np.nan
    return a


def _da(arr, dims=('y', 'x')):
    y, x = dims
    return xr.DataArray(
        arr, dims=(y, x),
        coords={y: np.arange(arr.shape[0]), x: np.arange(arr.shape[1])},
    )


def _assert_bitexact(got, exp):
    got, exp = np.asarray(got), np.asarray(exp)
    assert got.dtype == exp.dtype, f'dtype mismatch: {got.dtype} != {exp.dtype}'
    assert got.shape == exp.shape, f'shape mismatch: {got.shape} != {exp.shape}'
    assert np.array_equal(got, exp, equal_nan=True), 'values differ (bit-exactness broken)'


# ---------------------------------------------------------------------------
# spatial_resample upsample branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('shape,out,method,chunks', [
    ((100, 137), (200, 274), 'linear', {'y': 50, 'x': 68}),   # ratio 2, aligned chunks
    ((100, 137), (250, 343), 'linear', {'y': 40, 'x': 60}),   # fractional ratios, uneven last block
    ((100, 137), (150, 274), 'linear', {'y': 45, 'x': 70}),   # non-square, mixed ratios
    ((64, 64), (384, 384), 'linear', {'y': 100, 'x': 100}),   # ratio 6
    ((97, 97), (582, 582), 'repeat', {'y': 100, 'x': 100}),   # nearest, ratio 6
    ((50, 80), (100, 160), 'repeat', {'y': 33, 'x': 47}),     # nearest, non-square, uneven chunks
])
def test_upsample_bitexact(shape, out, method, chunks):
    arr = _da(_src(*shape))
    output_shape = dict(zip(('y', 'x'), out))

    res = spatial_resample(arr, output_shape, dict(chunks), method)
    ref = _sync_upsample_reference(arr, output_shape, dict(chunks), method)

    # Historical convention of this branch: swapped dim order, no coordinates.
    assert tuple(res.dims) == ('x', 'y')
    assert len(res.coords) == 0
    assert res.attrs == arr.attrs
    _assert_bitexact(res.compute().values, ref.compute().values)


@pytest.mark.parametrize('dtype,method,expected', [
    (np.float64, 'linear', np.float64),
    (np.float32, 'linear', np.float64),
    (np.int16, 'repeat', np.int16),
    (np.float32, 'repeat', np.float32),
])
def test_upsample_dtype(dtype, method, expected):
    arr = _da(_src(40, 60, dtype=dtype, nan_patch=False))
    res = spatial_resample(arr, {'y': 80, 'x': 120}, {'y': 30, 'x': 45}, method)
    assert res.dtype == expected


def test_upsample_lazy_and_local():
    """Construction reads nothing; computing one output block reads ~one source block."""
    ny, nx = 120, 160
    base = _src(ny, nx, seed=3)
    calls = []

    def read(block):
        calls.append(1)
        return block

    blocks = []
    for i in range(0, ny, 40):
        row = []
        for j in range(0, nx, 50):
            blk = base[i:i + 40, j:j + 50].copy()
            row.append(da.from_delayed(dask.delayed(read)(blk), shape=blk.shape, dtype=blk.dtype))
        blocks.append(row)
    src = da.block(blocks)
    arr = xr.DataArray(src, dims=('y', 'x'),
                       coords={'y': np.arange(ny), 'x': np.arange(nx)})

    output_shape = {'y': 240, 'x': 320}
    res = spatial_resample(arr, output_shape, {'y': 60, 'x': 80})
    assert len(calls) == 0, 'construction must not read pixel data'

    # First output block of the (swapped) result maps to source rows [0, ~30),
    # cols [0, ~40): inside the first 40x50 source block only.
    corner = res.isel(x=slice(0, 80), y=slice(0, 60)).compute()
    total_blocks = (ny // 40) * (nx // 50)
    assert 1 <= len(calls) <= 2
    assert len(calls) < total_blocks, 'partial compute must stay local'

    # Full compute still matches the eager reference bit-for-bit.
    ref = _sync_upsample_reference(arr, output_shape, {'y': 60, 'x': 80})
    _assert_bitexact(res.compute().values, ref.compute().values)


# ---------------------------------------------------------------------------
# tiepoint_angles
# ---------------------------------------------------------------------------

def _angles_ds(ny, nx, nr, nc, kind, seed=0):
    """Synthetic dataset with a full-grid dummy var and two float32 tie vars."""
    rng = np.random.default_rng(seed)
    y, x = str(names.rows), str(names.columns)
    if kind == 'regular':
        rows = np.arange(nr) * 8.33
        cols = np.arange(nc) * 8.33
    else:
        # Like a real tie grid, the first position is exactly 0 (the target
        # linspace starts there); only interior spacings are jittered (< spacing,
        # so the coords stay strictly monotonic but no longer regular).
        rows = np.concatenate(([0.0], np.arange(1, nr) * 8.33 + rng.random(nr - 1) * 0.5))
        cols = np.concatenate(([0.0], np.arange(1, nc) * 8.33 + rng.random(nc - 1) * 0.5))

    ds = xr.Dataset({'payload': ((y, x), np.zeros((ny, nx)))})
    for k, name in enumerate((names.sza, names.vaa)):
        t = (rng.random((nr, nc)) * 180).astype('float32')
        t[2:5, 3 + k:7 + k] = np.nan
        ds[f'{name}_tie'] = (('tie_rows', 'tie_columns'), t)
    return ds.assign_coords(tie_rows=rows, tie_columns=cols)


@pytest.mark.parametrize('kind', ['regular', 'irregular'])
@pytest.mark.parametrize('chunks', [None, {'y': 64, 'x': 128}])
def test_tiepoint_angles_bitexact(kind, chunks):
    ny, nx, nr, nc = 250, 350, 9, 13
    angles = (names.sza, names.vaa)
    ds = _angles_ds(ny, nx, nr, nc, kind)

    got = tiepoint_angles(ds.copy(), angles=angles, chunks=chunks)
    ref = _sync_angles_reference(ds.copy(), angles, chunks=chunks)

    for name in angles:
        g = got[str(name)]
        e = ref[str(name)]
        assert tuple(g.dims) == (str(names.rows), str(names.columns))
        assert g.dtype == np.float32
        _assert_bitexact(g.values, e.values)


def test_tiepoint_angles_lazy():
    """Construction reads nothing; one output block stays local to the tie grid."""
    ny, nx, nr, nc = 250, 350, 9, 13
    ds = _angles_ds(ny, nx, nr, nc, 'regular')
    angles = (names.sza,)

    calls = []
    tie = ds['sza_tie'].values.copy()

    def read(block):
        calls.append(1)
        return block

    # Replace the tie variable with a 1-chunk delayed wrapper around the data.
    lazy = da.from_delayed(dask.delayed(read)(tie), shape=tie.shape, dtype=tie.dtype)
    ds['sza_tie'] = (('tie_rows', 'tie_columns'), lazy)

    got = tiepoint_angles(ds, angles=angles, chunks={'y': 64, 'x': 128})
    assert len(calls) == 0, 'construction must not read the tie grid'

    # One full output chunk: the target positions it covers span only part of
    # the (tiny) tie grid, but the delayed wrapper is a single block, so at
    # most one read may happen.
    corner = got['sza'].isel(y=slice(0, 64), x=slice(0, 128)).values
    assert len(calls) <= 1

    ref = _sync_angles_reference(ds.copy(), angles, chunks={'y': 64, 'x': 128})
    _assert_bitexact(got['sza'].values, ref['sza'].values)
