#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from matplotlib import pyplot as plt
from eoread.sample_products import product_getter
from eoread.olci import Level1_OLCI, Level2_OLCI, olci_init_spectral
from eoread.olci import get_valid_l2_pixels
from eoread import eo
import numpy as np
from . import generic
from .generic import param, indices, scheduler
from .conftest import savefig


product = pytest.fixture(params=['prod_S3_L1_20190430'])(product_getter)


def test_olci_level1(product):
    ds = Level1_OLCI(product['path'])

    ds = eo.chunk(ds, bands=-1)
    ds.chunks    # check that it returned valid chunks

    # test method contains
    lat = ds.latitude[100, 100]
    lon = ds.longitude[100, 100]
    assert eo.contains(ds, lat, lon)
    assert not eo.contains(ds, lat, lon+180)

    assert 'total_ozone' in ds
    assert 'sea_level_pressure' in ds
    assert 'total_columnar_water_vapour' in ds


def test_split_merge(product):
    ds = Level1_OLCI(product['path'])
    print(ds)
    ds = eo.sub_rect(ds, 55, 56, 18, 19)
    split = eo.split(ds, 'bands')
    print(split)
    merge = eo.merge(split)
    print(merge)


def test_sub_pt(product):
    ds = Level1_OLCI(product['path'])
    lat0 = ds.latitude[500, 500]
    lon0 = ds.longitude[500, 500]
    eo.sub_pt(ds, lat0, lon0, 3)


@pytest.mark.parametrize('product', ['prod_S3_L2_20190612'], indirect=True)
def test_olci_level2(product):
    l2 = Level2_OLCI(product['path'])
    print(l2)


@pytest.mark.parametrize('product', ['prod_S3_L2_20190612'], indirect=True)
def test_olci_level2_flags(product):
    l2 = Level2_OLCI(product['path'])

    eo.getflags(l2.wqsf)
    get_valid_l2_pixels(l2.wqsf)


def test_main(product):
    ds = Level1_OLCI(product['path'])
    eo.init_Rtoa(ds)
    generic.test_main(ds)

def test_read(product, param, indices, scheduler):
    ds = Level1_OLCI(product['path'])
    eo.init_Rtoa(ds)
    generic.test_read(ds, param, indices, scheduler)


def test_subset(product):
    ds = Level1_OLCI(product['path'])
    generic.test_subset(ds)

@pytest.mark.parametrize('s', [
    slice(None, 300),
    slice(None, None, 10),
    ])
@pytest.mark.parametrize('interp', ['legacy', 'linear', 'atan2'])
def test_preview(product, param, request, s, interp):
    """
    Generate browses of various products
    """
    plt.figure()
    l1 = Level1_OLCI(product['path'], interp_angles=interp)
    eo.init_Rtoa(l1)
    if l1[param].ndim == 2:
        plt.imshow(l1[param][s, s])
    elif l1[param].ndim == 3:
        plt.imshow(l1[param][-1, s, s])
    plt.colorbar()
    savefig(request)


def test_flag(product):
    """
    Check that flags are properly raised
    """
    l1 = Level1_OLCI(product['path'])
    assert (l1.quality_flags > 0).any()
    assert (l1.flags > 0).any()
