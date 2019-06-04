#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pytest
import os
from eoread.level1_olci import Level1_OLCI
from eoread.level1_msi import Level1_MSI
from tests import products as p
from tests.products import sentinel_product, sample_data_path


@pytest.mark.parametrize('prod_id,prod_name', [p.prod_S3_20190430])
def test_olci(sentinel_product, capsys):
    ds = Level1_OLCI(sentinel_product)
#     with capsys.disabled():
    print(ds)


@pytest.mark.parametrize('prod_id,prod_name,resolution',
                         [p.prod_S2_20190419+(res,) for res in ['20', '60']])   # FIXME: 10m leads to memory error
def test_msi(sentinel_product, capsys, resolution):
    ds = Level1_MSI(sentinel_product, resolution)
    print(ds)
