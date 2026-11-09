#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import numpy as np
import eoread.eo as eo
from core.geo.naming import names

from . import generic
from eoread.msi import get_sample, Level1_MSI, Level2_MSI, FlagsReader_MSI
from eoread.flags import FlagsInit, GenericFlags


@pytest.fixture(scope="session")
def level1_msi(): 
    return get_sample(1)

@pytest.fixture(scope="session")
def level2_msi(): 
    return get_sample(2)

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture(params=[None, 10])
def resolution(request):
    return request.param

@pytest.fixture
def S2_product(level1_msi, chunks, resolution):
    return Level1_MSI(level1_msi, chunks=chunks, resolution=resolution)


@pytest.fixture(scope="session")
def l1_angle_default(level1_msi):
    ds = Level1_MSI(level1_msi, resolution=60, verbose=False)
    eo.init_geometry(ds)
    return ds

@pytest.fixture(scope="session")
def l1_angle_band2(level1_msi):
    ds = Level1_MSI(level1_msi, resolution=60, angle_band=2, verbose=False)
    eo.init_geometry(ds)
    return ds

@pytest.fixture(scope="session")
def l1_angle_all(level1_msi):
    ds = Level1_MSI(level1_msi, resolution=60, angle_band='all', verbose=False)
    eo.init_geometry(ds)
    return ds


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_msi, resolution, chunks):
    Level1_MSI(level1_msi, chunks=chunks, resolution=resolution)

def test_main(level1_msi, chunks):
    l1 = Level1_MSI(level1_msi, chunks=chunks, resolution=60).compute()
    generic.Test.main(l1, angle_data=True)

def test_angle_default_2d(l1_angle_default):
    """Default (angle_band=None) reads band-0 geometry as 2D fields (unchanged)."""
    ds = l1_angle_default
    for v in (names.sza, names.saa, names.vza, names.vaa, names.raa):
        assert ds[str(v)].dims == (str(names.rows), str(names.columns))

def test_angle_single_band(l1_angle_band2, level1_msi):
    """A single band selected by id or name yields the same 2D geometry."""
    ds_int = l1_angle_band2
    band_name = ds_int[str(names.bands)].values[2]
    ds_name = Level1_MSI(level1_msi, resolution=60, angle_band=band_name, verbose=False)
    eo.init_geometry(ds_name)
    for v in (names.vza, names.vaa, names.raa):
        assert ds_int[str(v)].dims == (str(names.rows), str(names.columns))
        assert np.allclose(ds_int[str(v)], ds_name[str(v)], equal_nan=True)

def test_angle_all_bands(l1_angle_all, l1_angle_band2):
    """angle_band='all' yields 3D per-band viewing geometry; solar angles stay 2D."""
    ds = l1_angle_all
    bands, rows, cols = str(names.bands), str(names.rows), str(names.columns)
    assert ds[str(names.vza)].dims == (bands, rows, cols)
    assert ds[str(names.vaa)].dims == (bands, rows, cols)
    assert ds[str(names.sza)].dims == (rows, cols)
    assert ds[str(names.saa)].dims == (rows, cols)
    assert ds[str(names.vza)].shape[0] == len(ds[bands])
    assert np.isfinite(ds[str(names.vza)].compute().values).any()
    # each band must match the corresponding single-band read
    band_name = ds[bands].values[2]
    assert np.allclose(
        ds[str(names.vza)].sel({bands: band_name}),
        l1_angle_band2[str(names.vza)],
        equal_nan=True,
    )

def test_angle_all_flags(l1_angle_all, l1_angle_default):
    """With per-band geometry, L1_INVALID stays 2D: invalid if vza is NaN in any band."""
    rows, cols = str(names.rows), str(names.columns)
    rd = FlagsReader_MSI()
    f_all = rd.getflag(l1_angle_all, GenericFlags.L1_INVALID).compute()
    f_def = rd.getflag(l1_angle_default, GenericFlags.L1_INVALID).compute()
    assert f_all.dims == f_def.dims == (rows, cols)
    # pixels invalid in band 0 are also invalid with all bands
    assert bool((f_all | ~f_def).all())

    # FlagsInit creates a 2D flags variable
    sub = l1_angle_all.isel({rows: slice(0, 200), cols: slice(0, 200)})
    flags = FlagsInit(
        flags={GenericFlags.L1_INVALID: 4},
        dtype='uint16',
        flag_reader=sub.attrs['_flag_reader'],
    ).map_blocks(sub)['flags']
    assert flags.dims == (rows, cols)


def test_angle_invalid(level1_msi):
    with pytest.raises(ValueError):
        Level1_MSI(level1_msi, resolution=60, angle_band=99, verbose=False)
    with pytest.raises(ValueError):
        Level1_MSI(level1_msi, resolution=60, angle_band='B99', verbose=False)

def test_time(level1_msi, chunks): 
    params = {'dirname': level1_msi, 'chunks': chunks}
    generic.Test.execution_time(Level1_MSI, params)
    
def test_lazy_load(S2_product):
    generic.Test.lazy_load(S2_product)

@pytest.mark.skip()
@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(S2_product, param, indices, scheduler):
    eo.init_geometry(S2_product)
    generic.Test.read(S2_product, param, indices, scheduler)

def test_subset(level1_msi, chunks): 
    l1 = Level1_MSI(level1_msi, chunks=chunks, metadata_template=[])
    generic.Test.subset(l1)

def test_plot(request, level1_msi):
    l1 = Level1_MSI(level1_msi, resolution=60)
    generic.plot(request, l1, 'B4', poi={"x": 1000, "y": 1000})
    
def test_flag_reader(S2_product):
    generic.Test.flagreader(S2_product)

def test_ancillary(level1_msi):
    """Embedded ECMWFT ancillary data is read and lazily interpolated to the tile grid."""
    aux_file = next((level1_msi/'GRANULE').glob('*/AUX_DATA/AUX_ECMWFT'), None)
    if aux_file is None:
        pytest.skip('sample product does not contain embedded ECMWFT ancillary data')
    
    l1 = Level1_MSI(level1_msi, chunks=500, resolution=60)
    dims = (str(names.rows), str(names.columns))
    expected = {
        'horizontal_wind': ('m s**-1', 0., np.inf),
        'total_column_ozone': ('DU', 200., 500.),
        'sea_level_pressure': ('hPa', 900., 1100.),
        'total_column_water_vapour': ('kg m**-2', 0., np.inf),
        'relative_humidity': ('%', 0., 100.),
    }
    for var, (unit, vmin, vmax) in expected.items():
        assert var in l1.data_vars, f'{var} is missing from the dataset'
        assert l1[var].dims == dims
        assert l1[var].attrs[str(names.unit)] == unit
        # lazily loaded
        assert l1[var].chunks is not None
        values = l1[var].compute().values
        # tile edges can lie outside the local ancillary grid
        finite = values[np.isfinite(values)]
        assert finite.size > 0
        assert finite.min() >= vmin and finite.max() <= vmax


def test_l2_instantiation(level2_msi, chunks):
    Level2_MSI(level2_msi, chunks=chunks, resolution=60)