#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tempfile
import numpy as np
import pytest
from eoread.olci import Level1_OLCI
from eoread.msi import Level1_MSI
from tests import products as p
from tests.products import sentinel_product, sample_data_path
from eoread.process import blockwise_method
from dask.diagnostics import ProgressBar

class Calib:
    def __init__(self, bands):
        self.calib = {
            400 : 1.000000, 412 : 1.000000,
            443 : 0.999210, 490 : 0.984674,
            510 : 0.986406, 560 : 0.988970,
            620 : 0.991810, 665 : 0.987590,
            674 : 1.000000, 681 : 1.000000,
            709 : 1.000000, 754 : 1.000000,
            760 : 1.000000, 764 : 1.000000,
            767 : 1.000000, 779 : 1.000000,
            865 : 1.000000, 885 : 1.000000,
            900 : 1.000000, 940 : 1.000000,
            1020: 1.000000,
        }
        self.coeff = np.array([self.calib[b] for b in bands.values]).reshape((-1, 1, 1))
    @blockwise_method(
        dims_blockwise=('rows', 'columns'),
        dims_out=[('bands', 'rows', 'columns')],
        dtypes=['float64'])
    def calc(self, Rtoa):
        return Rtoa * self.coeff

@pytest.mark.parametrize('product', [p.prod_S3_L1_20190430])
def test_processing(sentinel_product):
    ds = Level1_OLCI(sentinel_product, init_reflectance=True)

    ds['Rtoa_calib'] = Calib(ds.bands).calc(ds.Rtoa)

    sub = ds.isel(
        rows=slice(1000, 1100),
        columns=slice(700, 750),
    )
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpf, ProgressBar():
        sub.to_netcdf(tmpf.name)
