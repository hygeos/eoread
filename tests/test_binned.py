#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from eoread.sample_products import get_sample_products
from eoread.utils.binned import read_binned, ncols, Binner, to_2dim
from . import conftest

p = get_sample_products()

products = [
    (p['prod_MODIS_binned_chl'], 'chl_ocx'),
]

@pytest.mark.parametrize('product,varname', products)
def test_read_binned(product, varname, request):

    dat = read_binned(product['path'], varname)[0]
    dat[np.isnan(dat)] = 0.

    plt.figure()
    plt.imshow(dat,
        vmin=0,
        vmax=dat.mean()*1.4,
        )
    plt.colorbar()
    conftest.savefig(request)


@pytest.mark.parametrize('product', [p[0] for p in products])
def test_check_ncols(product):
    ds = xr.open_dataset(
        product['path'],
        group='level-3_binned_data',
        )
    neq = len(ds.binIndexDim)*2

    start_num = ds.BinIndex.data['start_num']
    nc = ncols(neq)
    assert (np.diff(start_num) == nc[:-1]).all()

@pytest.mark.parametrize('product,varname', products)
def test_consistency(product, varname, request):
    dat, lat, lon = read_binned(product['path'], varname)

    ok = ~np.isnan(dat)
    neq = 2*len(xr.open_dataset(product['path'], group='level-3_binned_data').BinIndex)

    binner = Binner(neq)
    binner.add(dat[ok], lat[ok], lon[ok])

    dat1, lat1, lon1 = to_2dim(binner.values(), neq)

    np.testing.assert_allclose(dat, dat1)

    for (txt, d) in [
            ('data', dat),
            ('lat', lat),
            ('lon', lon),
            ('data1', dat1),
            ('lat1', lat1),
            ('lon1', lon1),
        ]:
        plt.figure()
        plt.imshow(d)
        plt.colorbar()
        plt.title(txt)
        conftest.savefig(request)
