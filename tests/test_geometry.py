#!/usr/bin/env python3
"""
Bit-exact tests for the lazy UTM->lat/lon transform (eoread.transforms.geometry).

The lazy implementation must reproduce the former full-grid computation exactly:
same float64 (X, Y) inputs per point, same pointwise pyproj call, same final
float32 rounding. Verified against a verbatim copy of the old eager code.
"""

import dask.array as da
import numpy as np
import pytest
from pyproj import Proj

from core.geo.naming import names
from eoread.transforms.geometry import latlon_utm

CRS = 'EPSG:32631'
ULX, ULY, XDIM, YDIM = 600000.0, 7500040.0, 10.0, 10.0


def _eager_reference(nx: int, ny: int):
    """Verbatim eager implementation (pre-lazy rewrite), returns (lon, lat)."""
    proj = Proj(CRS)
    x = ULX + XDIM / 2 + XDIM * np.arange(nx)
    y = ULY + YDIM / 2 + YDIM * np.arange(ny)
    X, Y = np.meshgrid(x, y)
    lon, lat = proj(X, Y, inverse=True)
    return lon.astype('float32'), lat.astype('float32')


@pytest.mark.parametrize('nx, ny', [(600, 500), (500, 600)])
@pytest.mark.parametrize('chunks', [
    None,
    {str(names.rows): 500, str(names.columns): 500},
    {str(names.rows): 333, str(names.columns): 177},
])
def test_bitexact_vs_eager(nx, ny, chunks):
    lon_ref, lat_ref = _eager_reference(nx, ny)
    res = latlon_utm(CRS, ULX, ULY, XDIM, YDIM, nx, ny, chunks)
    np.testing.assert_array_equal(res[str(names.lon)].values, lon_ref)
    np.testing.assert_array_equal(res[str(names.lat)].values, lat_ref)


def test_dtype_dims_and_layout():
    res = latlon_utm(CRS, ULX, ULY, XDIM, YDIM, 500, 600,
                     {str(names.rows): 250, str(names.columns): 200})
    for var in (str(names.lon), str(names.lat)):
        arr = res[var]
        assert arr.dims == (str(names.rows), str(names.columns))
        assert arr.dtype == np.float32
        assert len(arr.coords) == 0
        assert isinstance(arr.data, da.Array)
    assert res[str(names.lon)].chunks == ((250, 250, 100), (200, 200, 100))


def test_single_block_when_chunks_none():
    res = latlon_utm(CRS, ULX, ULY, XDIM, YDIM, 300, 200, None)
    assert res[str(names.lon)].chunks == ((200,), (300,))


def test_lazy_no_full_grid_in_graph():
    """Construction must not materialize pixel-sized arrays.

    The former eager implementation built full (ny, nx) float64 X/Y meshes at
    construction time; the lazy one holds only the 1-D coordinate vectors in
    the graph and transforms per block at compute time (plus one empty-array
    meta-inference probe by dask itself). Assert that no array value in the
    graph reaches pixel size.

    Note: counting transform calls via monkeypatched helpers is unreliable
    with dask >= 2026.1 — its tokenize() does a pickle round-trip of
    non-importable (test-defined) objects, re-creating their classes and
    functions with fresh globals that silently redirect any test counters.
    """
    nx, ny = 600, 500
    res = latlon_utm(CRS, ULX, ULY, XDIM, YDIM, nx, ny,
                     {str(names.rows): 200, str(names.columns): 200})
    for var in (str(names.lon), str(names.lat)):
        for value in res[var].data.__dask_graph__().values():
            if isinstance(value, np.ndarray):
                assert value.size < nx * ny, \
                    'pixel-sized array materialized at construction time'
