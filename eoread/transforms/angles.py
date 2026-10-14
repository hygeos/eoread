#!/usr/bin/env python3
"""
Tie-point angle transforms shared by v2 readers.
"""

import numpy as np
import xarray as xr

from core.geo.naming import names

from eoread.tools import _blockwise_interp


def tiepoint_angles(ds: xr.Dataset, angles=(names.sza, names.saa, names.vza, names.vaa),
                    tie_dims=('tie_rows', 'tie_columns'), chunks=None) -> xr.Dataset:
    """
    Interpolate 2D tie-point angle variables onto the full (y, x) grid.

    For each angle `name`, expects a variable `str(name) + '_tie'` with dims
    `tie_dims` and 1D coordinate arrays of the same names holding the pixel
    positions of the tie points. Adds the full-grid variable `str(name)`
    (float32, dims (y, x)).

    The interpolation is blockwise-lazy: each output chunk is interpolated from
    a small window of the tie-point grid (the target grid spans exactly the
    tie-point range), so no full-grid data is materialized at construction.
    Results are bit-for-bit identical to a full-grid
    `core.interpolate.interp` evaluation.

    Parameters
    ----------
    ds : Dataset
        Must carry the full-grid dimensions y/x and the `*_tie` variables with
        index coordinates on `tie_dims`.
    angles : sequence of registry names
        Angles to interpolate (default: sza, saa, vza, vaa).
    tie_dims : tuple of str
        Dim names of the tie-point variables (row dim first).
    chunks : dict or None
        Chunk sizes applied to the full-grid results.

    Returns
    -------
    The input dataset with the full-grid angle variables added.
    """
    ny = ds.sizes[str(names.rows)]
    nx = ds.sizes[str(names.columns)]
    tie_r, tie_c = tie_dims
    t_rows = np.asarray(ds[tie_r].values)
    t_cols = np.asarray(ds[tie_c].values)
    y_pos = np.linspace(0, float(t_rows[-1]), ny)
    x_pos = np.linspace(0, float(t_cols[-1]), nx)
    c_rows = chunks[str(names.rows)] if chunks is not None else ny
    c_cols = chunks[str(names.columns)] if chunks is not None else nx

    out = {}
    for name in angles:
        tie = ds[f'{name}_tie']
        arr = _blockwise_interp(tie.data, (t_rows, t_cols), (y_pos, x_pos),
                                (c_rows, c_cols), 'linear')
        var = xr.DataArray(arr, dims=(str(names.rows), str(names.columns)), attrs=tie.attrs)
        out[str(name)] = var.astype('float32')
    return ds.assign(out)
