#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the ancillary data providers.

Covers the shared date helpers (`common.floor_dt`/`ceil_dt`), the NASA GMAO
resource functions, `wrap_lon`, and that the ERA5 provider imports without
the optional `cdsapi` dependency (it must stay lazy). No network access.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------- date helpers

def test_floor_ceil_dt():
    from eoread.ancillary.era5 import ceil_dt, floor_dt

    dt = datetime(2020, 1, 2, 15, 37, 22)
    d3 = timedelta(hours=3)
    assert floor_dt(dt, d3) == datetime(2020, 1, 2, 15, 0)
    assert ceil_dt(dt, d3) == datetime(2020, 1, 2, 18, 0)

    # exact period boundary: both round to the boundary itself (the bracketing
    # interval collapses to a single file)
    b = datetime(2020, 1, 2, 15, 0)
    assert floor_dt(b, d3) == b
    assert ceil_dt(b, d3) == b

    # hour grid
    assert floor_dt(dt, timedelta(hours=1)) == datetime(2020, 1, 2, 15, 0)
    assert ceil_dt(dt, timedelta(hours=1)) == datetime(2020, 1, 2, 16, 0)


# ------------------------------------------------------------------- NASA GMAO

def _import_nasa_anc():
    from eoread.ancillary import nasa
    return nasa


def test_resource_gmao_fp():
    from hygeos_core.dates import round_date
    anc = _import_nasa_anc()

    dt = datetime(2020, 1, 2, 15, 37, 22)
    d0, d1 = round_date(dt, 3)
    out = anc.resource_gmao_fp(dt)

    # one (pattern, date) pair per bracketing date; the pattern is a strftime
    # template applied to each date by Ancillary_NASA.download
    assert [d for _, d in out] == [d0, d1]
    assert [d.strftime(out[0][0]) for d in (d0, d1)] == \
        ['GMAO_FP.20200102T150000.MET.NRT.nc', 'GMAO_FP.20200102T180000.MET.NRT.nc']

    # the bracketing dates surround dt at a 3-hour period
    assert d0 <= dt <= d1
    assert d1 - d0 == timedelta(hours=3)


def test_resource_gmao_merra2():
    from hygeos_core.dates import round_date
    anc = _import_nasa_anc()

    dt = datetime(2020, 1, 2, 15, 37, 22)
    d0, d1 = round_date(dt, 1)
    out = anc.resource_gmao_merra2(dt)

    assert [d for _, d in out] == [d0, d1]
    assert [d.strftime(out[0][0]) for d in (d0, d1)] == \
        ['GMAO_MERRA2.20200102T150000.MET.nc', 'GMAO_MERRA2.20200102T160000.MET.nc']


def test_wrap_lon():
    anc = _import_nasa_anc()

    lons = np.arange(-180., 180.)
    da = xr.DataArray(
        np.arange(lons.size, dtype='f4').reshape(1, -1),
        dims=('latitude', 'longitude'),
        coords={'latitude': [0.], 'longitude': lons},
    )
    out = anc.wrap_lon(da)

    # the first column is duplicated at longitude 180 (size + 1, same values)
    assert out.sizes['longitude'] == da.sizes['longitude'] + 1
    assert float(out.longitude.values[-1]) == 180.
    np.testing.assert_array_equal(
        out.isel(longitude=0).values, out.isel(longitude=-1).values)


# ------------------------------------------------------------------------ ERA5

def test_era5_imports_without_cdsapi():
    # cdsapi is an optional dependency (pyproject `era5` extra): importing the
    # provider must not require it; only download_file needs a client.
    from eoread.ancillary import era5

    try:
        import cdsapi  # noqa: F401
        pytest.skip('cdsapi installed; the lazy-import path is not exercised')
    except ImportError:
        pass  # cdsapi absent — the module still imported fine
    assert hasattr(era5, 'ERA5')
