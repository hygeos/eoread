#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from tests.products import products as p, get_path
from eoread.olci import Level1_OLCI, Level2_OLCI, olci_init_spectral
from eoread.olci import get_valid_l2_pixels
from eoread import eo
from . import generic
from .generic import param, indices


@pytest.mark.parametrize('product', [p['prod_S3_L1_20190430']])
def test_olci_level1(product):
    ds = Level1_OLCI(get_path(product))

    # test method contains
    lat = ds.latitude[100, 100]
    lon = ds.longitude[100, 100]
    assert eo.contains(ds, lat, lon)
    assert not eo.contains(ds, lat, lon+180)

    assert 'total_ozone' in ds
    assert 'sea_level_pressure' in ds
    assert 'total_columnar_water_vapour' in ds


@pytest.mark.parametrize('product', [p['prod_S3_L1_20190430']])
def test_split_merge(product):
    ds = Level1_OLCI(get_path(product))
    print(ds)
    ds = eo.sub_rect(ds, 55, 56, 18, 19)
    split = eo.split(ds, 'bands')
    print(split)
    merge = eo.merge(split)
    print(merge)


@pytest.mark.parametrize('product', [p['prod_S3_L1_20190430']])
def test_sub_pt(product):
    ds = Level1_OLCI(get_path(product))
    lat0 = ds.latitude[500, 500]
    lon0 = ds.longitude[500, 500]
    eo.sub_pt(ds, lat0, lon0, 3)


@pytest.mark.parametrize('product', [p['prod_S3_L2_20190612']])
def test_olci_level2(product):
    l2 = Level2_OLCI(get_path(product))
    print(l2)


@pytest.mark.parametrize('product', [p['prod_S3_L2_20190612']])
def test_olci_level2_flags(product):
    l2 = Level2_OLCI(get_path(product))

    eo.getflags(l2.wqsf)
    get_valid_l2_pixels(l2.wqsf)


@pytest.mark.parametrize('product', [p['prod_S3_L1_20190430']])
def test_main(product):
    ds = Level1_OLCI(get_path(product))
    eo.init_Rtoa(ds)
    generic.test_main(ds)

@pytest.mark.parametrize('product', [p['prod_S3_L1_20190430']])
def test_read(product, param, indices):
    ds = Level1_OLCI(get_path(product))
    eo.init_Rtoa(ds)
    generic.test_read(ds, param, indices)


@pytest.mark.parametrize('product', [p['prod_S3_L1_20190430']])
def test_subset(product):
    ds = Level1_OLCI(get_path(product))
    generic.test_subset(ds)
