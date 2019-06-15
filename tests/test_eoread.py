#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from eoread.olci import Level1_OLCI, Level2_OLCI
from eoread.msi import Level1_MSI
from tests import products as p
from tests.products import sentinel_product, sample_data_path
from eoread.common import GeoDatasetAccessor


@pytest.mark.parametrize('product', [p.prod_S3_L1_20190430])
def test_olci_level1(sentinel_product, capsys):
    ds = Level1_OLCI(sentinel_product)
#     with capsys.disabled():
    print(ds)

    # test method contains
    lat = ds.latitude[100, 100]
    lon = ds.longitude[100, 100]
    assert ds.eo.contains(lat, lon)
    assert not ds.eo.contains(lat, lon+180)


@pytest.mark.parametrize('product', [p.prod_S3_L2_20190612])
def test_olci_level2(sentinel_product, capsys):
    ds = Level2_OLCI(sentinel_product)
    print(ds)


@pytest.mark.parametrize('product,resolution',
                         [(p.prod_S2_L1_20190419,res) for res in ['20', '60']])   # FIXME: 10m leads to memory error
def test_msi(sentinel_product, capsys, resolution):
    ds = Level1_MSI(sentinel_product, resolution)
    print(ds)
