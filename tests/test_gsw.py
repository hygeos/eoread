#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from matplotlib import pyplot as plt
from eoread.reader.gsw import GSW, read_tile
from tempfile import TemporaryDirectory
from eoread.sample_products import get_sample_products
from eoread.reader.olci import Level1_OLCI, get_sample
from . import conftest

p = get_sample_products()

@pytest.mark.parametrize('agg', [4, 8])
def test_single_tile(request, agg):
    with TemporaryDirectory() as tmpdir:
        T = read_tile('0E_50N', agg, directory=tmpdir)
        print(T)
        assert T.shape == T.compute().shape
        plt.imshow(T[::10, ::10])
        conftest.savefig(request)


@pytest.mark.parametrize('agg', [1, 2, 4, 8])
def test_gsw_instantiate(agg):
    gsw = GSW(agg=agg)
    assert gsw.dims == ('latitude', 'longitude')
    print(gsw)


def test_gsw_zoom(request):
    gsw = GSW(agg=8)
    print(gsw)

    sub = gsw.where(
        (gsw.latitude > 38.628122)
        & (gsw.latitude < 41.456405)
        & (gsw.longitude > 7.574776)
        & (gsw.longitude < 10.115557),
        drop=True,
    )
    print(sub)
    sub.plot()
    conftest.savefig(request)


def test_index(request):
    """
    Check whether we can do fancy indexing
    """
    product = get_sample('level1_fr')
    l1 = Level1_OLCI(product)
    l1 = l1.isel(
        y=slice(None, None, 10),
        x=slice(None, None, 10))

    plt.figure()
    plt.imshow(l1.Ltoa.sel(bands=865))
    conftest.savefig(request)

    gsw = GSW(agg=8)

    # water mask
    mask = gsw.sel(
        latitude=l1.latitude,
        longitude=l1.longitude,
        method='nearest') < 50
    print(mask)
    mask = mask.compute(scheduler='sync')
    plt.imshow(mask)
    conftest.savefig(request)
