#!/usr/bin/env python3
"""
Geometry transforms shared by v2 readers.

One implementation per operation (REFACTORING.md, §3): readers call these
functions instead of carrying per-module copies.
"""

import numpy as np
from pyproj import Proj
import xarray as xr

from core.geo.naming import names


def latlon_utm(crs: str, ulx: float, uly: float, xdim: float, ydim: float,
               nx: int, ny: int, chunks=None) -> dict:
    """
    Compute (latitude, longitude) rasters from UTM tile geocoding metadata.

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
    X, Y = np.meshgrid(x, y)
    lon, lat = proj(X, Y, inverse=True)
    dims = (str(names.rows), str(names.columns))
    return {
        str(names.lon): xr.DataArray(lon.astype('float32'), dims=dims).chunk(chunks),
        str(names.lat): xr.DataArray(lat.astype('float32'), dims=dims).chunk(chunks),
    }
