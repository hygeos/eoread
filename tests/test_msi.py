#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import numpy as np
from core.geo.naming import names

from . import generic
from eoread.msi import get_sample, Level1_MSI, Level2_MSI


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


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_msi, resolution, chunks):
    Level1_MSI(level1_msi, chunks=chunks, resolution=resolution)

def test_main(level1_msi, chunks):
    l1 = Level1_MSI(level1_msi, chunks=chunks, resolution=60).compute()
    generic.Test.main(l1, angle_data=True)
    
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