#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from eoread.reader.venus import Level1_VENUS, get_SRF
from eoread.utils.graphics import plot_srf
from . import generic
from eoread import eo
from . import conftest
from matplotlib import pyplot as plt

import pytest
import xarray as xr

product = Path('/archive2/data/VENUS/VENUS-XS_20230116-112657-000_L1C_VILAINE_C_V3-1/')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def S2_product(chunks):
    return Level1_VENUS(product, chunks=chunks)

@pytest.mark.parametrize('split', [True, False])
def test_instantiation(split, chunks):
    Level1_VENUS(product, split=split, chunks=chunks)


@pytest.mark.parametrize('param', ['sza', 'vza', 'saa', 'vaa', 'latitude', 'longitude'])
def test_msi_merged(S2_product, param):
    l1 = S2_product
    print(l1)
    assert 'Rtoa_420' not in l1
    assert 'Rtoa' in l1

    # check parameter consistency through windowing
    xr.testing.assert_allclose(
        l1[param][1000, 500],
        l1.isel(y=slice(1000, None),
                x=slice(500, None))[param][0, 0])

    xr.testing.assert_allclose(
        l1[param][1000:1010, 500:510],
        l1.isel(y=slice(1000, None),
                x=slice(500, None))[param][:10, :10])


@pytest.mark.parametrize('band', ['Rtoa_420', 'Rtoa_490', 'Rtoa_865'])
def test_msi_split(band):
    l1 = Level1_VENUS(product, split=True)
    print(l1)
    assert 'Rtoa_420' in l1
    assert 'Rtoa' not in l1

    assert l1[band][:10, :10].values.shape == (10, 10)

    xr.testing.assert_allclose(
            l1[band][:600, :600].compute()[500:550, 450:550],
            l1.sel(y=slice(500, 550), x=slice(450, 550))[band],
            )



def test_main(S2_product):
    generic.test_main(S2_product)


@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(S2_product, scheduler):
    eo.init_geometry(S2_product)
    generic.test_read(S2_product, 'Rtoa', (1000,1000), scheduler)


def test_subset(S2_product):
    generic.test_subset(S2_product)


def test_plot(request):
    l1 = Level1_VENUS(product)
    plt.imshow(
        l1.Rtoa.sel(bands=865),
        vmin=0, vmax=0.5)
    plt.colorbar()

    conftest.savefig(request)


def test_srf(request):
    srf = get_SRF()
    plot_srf(srf)
    conftest.savefig(request, bbox_inches="tight")
