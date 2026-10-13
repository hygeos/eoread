#!/usr/bin/env python3
"""
Tie-point angle transforms shared by v2 readers.
"""

import numpy as np
import xarray as xr

from core.geo.naming import names
from core.interpolate import Linear, interp


def tiepoint_angles(ds: xr.Dataset, angles=(names.sza, names.saa, names.vza, names.vaa),
                    tie_dims=('tie_rows', 'tie_columns'), chunks=None) -> xr.Dataset:
    """
    Interpolate 2D tie-point angle variables onto the full (y, x) grid.

    For each angle `name`, expects a variable `str(name) + '_tie'` with dims
    `tie_dims` and 1D coordinate arrays of the same names holding the pixel
    positions of the tie points. Adds the full-grid variable `str(name)`
    (float32, dims (y, x)).

    The tie-point grid is small; it is computed once (scheduler='sync') before
    interpolation. The output stays dask-backed when `chunks` is given.

    Parameters
    ----------
    ds : Dataset
        Must carry the full-grid dimensions y/x and the `*_tie` variables.
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
    y = xr.DataArray(np.linspace(0, float(ds[tie_r][-1].values), ny),
                     dims=(str(names.rows),)).chunk(-1)
    x = xr.DataArray(np.linspace(0, float(ds[tie_c][-1].values), nx),
                     dims=(str(names.columns),)).chunk(-1)

    out = {}
    for name in angles:
        tie = ds[f'{name}_tie']
        full = interp(tie.compute(scheduler='sync'), **{tie_r: Linear(y), tie_c: Linear(x)})
        arr = full.astype('float32')
        out[str(name)] = arr.chunk(chunks) if chunks is not None else arr
    return ds.assign(out)
