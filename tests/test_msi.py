#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
import xarray as xr
from eoread.download_legacy import download_S2_google, download_sentinelapi
from eoread.reader.msi import Level1_MSI, get_sample
from . import generic
from eoread import eo
from . import conftest
from matplotlib import pyplot as plt
from .generic import param, indices  # noqa

resolutions = ['10', '20', '60']


@pytest.fixture
def level1_msi() -> Path:
    return get_sample()

@pytest.fixture(params=resolutions)
def resolution(request):
    return request.param

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def S2_product(level1_msi, resolution, chunks):
    return Level1_MSI(level1_msi, resolution, chunks=chunks)


@pytest.mark.parametrize('split', [True, False])
def test_instantiation(level1_msi, resolution, split, chunks):
    Level1_MSI(level1_msi, resolution, split=split, chunks=chunks)


@pytest.mark.parametrize('param', ['sza', 'vza', 'saa', 'vaa', 'latitude', 'longitude'])
def test_msi_merged(S2_product, param):
    l1 = S2_product
    print(l1)
    assert 'Rtoa_443' not in l1
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

    if resolution == '60':
        full = l1[param].compute(scheduler='single-threaded')
        xr.testing.assert_allclose(
            full[1000:1010, 500:510],
            l1.isel(y=slice(1000, None),
                    x=slice(500, None))[param][:10, :10])


@pytest.mark.parametrize('band', ['Rtoa_443', 'Rtoa_490', 'Rtoa_865'])
def test_msi_split(level1_msi, band, resolution):
    l1 = Level1_MSI(level1_msi, resolution, split=True)
    print(l1)
    assert 'Rtoa_443' in l1
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
def test_read(S2_product, param, indices, scheduler):
    eo.init_geometry(S2_product)
    generic.test_read(S2_product, param, indices, scheduler)


def test_subset(S2_product):
    generic.test_subset(S2_product)


def test_plot(request, level1_msi):
    l1 = Level1_MSI(level1_msi)
    plt.imshow(
        l1.Rtoa.sel(bands=865),
        vmin=0, vmax=0.5)
    plt.colorbar()

    conftest.savefig(request)


@pytest.fixture(params=[
    {'product': 'S2B_MSIL2A_20190901T105619_N0213_R094_T30TWT_20190901T141237',
     'source': 'google'},
    {'product': 'S2A_MSIL2A_20230418T105621_N0509_R094_T31UCR_20230418T170158',
     'source': 'scihub'},
])
def level2_msi(request):
    dir_samples = Path(__file__).parent.parent/'SAMPLE_DATA'
    source = request.param['source']
    product = request.param['product']
    if source == 'google':
        return download_S2_google(product, dir_samples)
    elif source == 'scihub':
        target = dir_samples/product
        download_sentinelapi(target)
        return target
    else:
        raise ValueError


def test_level2(request, level2_msi: Path):
    assert level2_msi.exists()
    