#!/usr/bin/env python3
"""
Geometry transforms shared by v2 readers.

One implementation per operation (REFACTORING.md, §3): readers call these
functions instead of carrying per-module copies.
"""

import warnings

import dask.array as da
import numpy as np
from pyproj import Proj
import xarray as xr

from core.geo.naming import names


def latlon_utm(crs: str, ulx: float, uly: float, xdim: float, ydim: float,
               nx: int, ny: int, chunks=None) -> dict:
    """
    Compute (latitude, longitude) rasters from UTM tile geocoding metadata.

    The transformation is applied lazily, block by block: no pixel-sized array
    is materialized at construction time. Results are bit-for-bit identical to
    the former full-grid implementation — each output block transforms exactly
    the same float64 (X, Y) values the full meshgrid would have contained, and
    the UTM->lat/lon transform is a deterministic pointwise function.

    Parameters
    ----------
    crs : str
        UTM CRS code of the product grid, e.g. 'EPSG:32631'.
    ulx, uly : float
        Easting/northing (m) of the top-left corner of the grid.
    xdim, ydim : float
        Pixel size (m) in the X and Y directions.
    nx, ny : int
        Number of columns / rows of the target grid.
    chunks : dict or None
        Chunk sizes for the output rasters.

    Returns
    -------
    dict with 'latitude' and 'longitude' DataArrays, dims (y, x), float32.
    """
    proj = Proj(crs)
    x = ulx + xdim / 2 + xdim * np.arange(nx)
    y = uly + ydim / 2 + ydim * np.arange(ny)

    def _block(yb: np.ndarray, xb: np.ndarray) -> np.ndarray:
        # blockwise passes the dy block first, then the dx block.
        # np.meshgrid(xb, yb) holds exactly the (X, Y) values of this block in
        # the full-grid meshgrid — no arithmetic beyond the broadcast.
        X, Y = np.meshgrid(xb, yb)
        lon, lat = proj(X, Y, inverse=True)
        return np.stack([lon, lat]).astype('float32')

    c_rows = chunks[str(names.rows)] if chunks is not None else ny
    c_cols = chunks[str(names.columns)] if chunks is not None else nx
    dy = da.from_array(y, chunks=min(c_rows, ny))
    dx = da.from_array(x, chunks=min(c_cols, nx))
    # Output layout: 'k' is the new size-2 axis (lon/lat stacked), 'i' and
    # 'j' follow the dy/dx block axes. dask warns when a blockwise output
    # multiplies the input chunk count by >= 10 — here ~22x22 partitions from
    # two single-chunk 1-D vectors, which is exactly what this call exists for
    # (meshgrid-style expansion), so the warning is a false positive.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Increasing number of chunks')
        out = da.blockwise(
            _block, 'kij',
            dy, 'i', dx, 'j',
            new_axes={'k': 2},
            dtype='float32',
        )
    dims = (str(names.rows), str(names.columns))
    return {
        str(names.lon): xr.DataArray(out[0], dims=dims),
        str(names.lat): xr.DataArray(out[1], dims=dims),
    }


def latlon_bbox(north: float, south: float, west: float, east: float,
                nx: int, ny: int, chunks=None) -> dict:
    """
    Compute (latitude, longitude) rasters from a scene bounding box.

    For products whose grid is axis-aligned in geographic space — the row and
    column spacings span the bounding box to sub-pixel precision — the four
    corners pin the rasters: latitude is constant along columns and longitude
    along rows, each on an even linspace across the box. The 1-D vectors are built eagerly and
    tiled per block (same scheme as `latlon_utm`): no pixel-sized array is
    materialized at construction time.

    The linspaces run north->south / west->east, i.e. in the product's native
    north-up orientation; readers that emit south-up output flip the result
    together with the data (see the ECOSTRESS reader).

    Parameters
    ----------
    north, south, west, east : float
        Bounding box edges (degrees).
    nx, ny : int
        Number of columns / rows of the target grid.
    chunks : dict or None
        Chunk sizes for the output rasters.

    Returns
    -------
    dict with 'latitude' and 'longitude' DataArrays, dims (y, x), float32.
    """
    lat = np.linspace(north, south, ny)
    lon = np.linspace(west, east, nx)

    def _block(yb: np.ndarray, xb: np.ndarray) -> np.ndarray:
        # blockwise passes the dy block first, then the dx block.
        LON, LAT = np.meshgrid(xb, yb)
        return np.stack([LON, LAT]).astype('float32')

    c_rows = chunks[str(names.rows)] if chunks is not None else ny
    c_cols = chunks[str(names.columns)] if chunks is not None else nx
    dy = da.from_array(lat, chunks=min(c_rows, ny))
    dx = da.from_array(lon, chunks=min(c_cols, nx))
    # Output layout: 'k' is the new size-2 axis (lon/lat stacked), 'i' and
    # 'j' follow the dy/dx block axes. dask warns when a blockwise output
    # multiplies the input chunk count by >= 10 — here ~15x15 partitions from
    # two single-chunk 1-D vectors, which is exactly what this call exists for
    # (meshgrid-style expansion), so the warning is a false positive.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Increasing number of chunks')
        out = da.blockwise(
            _block, 'kij',
            dy, 'i', dx, 'j',
            new_axes={'k': 2},
            dtype='float32',
        )
    dims = (str(names.rows), str(names.columns))
    return {
        str(names.lon): xr.DataArray(out[0], dims=dims),
        str(names.lat): xr.DataArray(out[1], dims=dims),
    }
