#!/usr/bin/env python3
"""
Single-grid resampling for v2 readers.

A v2 dataset carries exactly one grid (dims y, x). Multi-resolution products
are normalized onto that grid at load time: fine bands are downsampled
lazily (coarsen), coarse bands are upsampled — no data is lost
(see REFACTORING.md, §5).
"""

import xarray as xr

from core.geo.naming import names
from eoread.tools import spatial_resample


def single_grid(ds: xr.Dataset, output_shape: dict, chunks: dict, method: str = 'linear') -> xr.Dataset:
    """
    Resample every raster of `ds` onto a single (y, x) grid.

    Handles 2D (y, x) rasters and 3D (bands, y, x) payloads; any other variable
    passes through untouched. Rasters already at the target shape are returned
    as-is.

    Parameters
    ----------
    ds : Dataset
        Rasters must use the canonical dim names/order.
    output_shape : dict
        Target sizes keyed by dim name, e.g. {'y': ny, 'x': nx}.
    chunks : dict
        Chunk sizes keyed by dim name (same keys as `output_shape`).
    method : str
        Upsampling method, 'linear' or 'repeat'.

    Returns
    -------
    The input dataset with its rasters resampled onto the target grid.
    """
    y, x = str(names.rows), str(names.columns)
    bands = str(names.bands)
    out_vars = {}
    for name, var in ds.data_vars.items():
        dims = set(var.dims)
        if dims == {y, x}:
            out_vars[name] = spatial_resample(var, output_shape, chunks, method)
        elif dims == {bands, y, x} and tuple(var.dims) == (bands, y, x):
            res_list = [
                spatial_resample(var.isel({bands: i}, drop=True), output_shape, chunks, method)
                for i in range(var.sizes[bands])
            ]
            out_vars[name] = xr.concat(res_list, dim=var[bands])
        # other variables pass through unchanged
    return ds.assign(out_vars)
