#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from matplotlib import pyplot as plt
from eoread.reader.olci import get_sample
from eoread.autodetect import Level1, Level2
from eoread.reader.olci import get_valid_l2_pixels
from eoread import eo
from core.tools import chunk, contains
from . import generic
from .generic import param, indices, scheduler  # noqa (fixtures)
from core.conftest import savefig


olci_level1 = pytest.fixture(lambda: get_sample('level1_fr'))
olci_level2 = pytest.fixture(lambda: get_sample('level2_fr'))


def test_olci_level1(olci_level1):
    ds = Level1(olci_level1)

    ds = chunk(ds, bands=-1)
    ds.chunks    # check that it returned valid chunks

    # test method contains
    lat = ds.latitude[100, 100]
    lon = ds.longitude[100, 100]
    assert contains(ds, lat, lon)
    assert not contains(ds, lat, lon+180)

    assert 'total_column_ozone' in ds
    assert 'sea_level_pressure' in ds
    assert 'total_columnar_water_vapour' in ds


def test_split_merge(olci_level1):
    ds = Level1(olci_level1)
    print(ds)
    ds = eo.sub_rect(ds, 55, 56, 18, 19)
    split = eo.split(ds, 'bands')
    print(split)
    merge = eo.merge(split)
    print(merge)


def test_sub_pt(olci_level1):
    ds = Level1(olci_level1)
    lat0 = ds.latitude[500, 500]
    lon0 = ds.longitude[500, 500]
    eo.sub_pt(ds, lat0, lon0, 3)


def test_olci_level2(olci_level2):
    l2 = Level2(olci_level2)
    print(l2)


def test_olci_level2_flags(olci_level2):
    l2 = Level2(olci_level2)

    eo.getflags(l2.wqsf)
    get_valid_l2_pixels(l2.wqsf)


def test_main(olci_level1):
    ds = Level1(olci_level1)
    eo.init_Rtoa(ds)
    generic.test_main(ds)

def test_read(olci_level1, param, indices, scheduler):
    ds = Level1(olci_level1)
    eo.init_Rtoa(ds)
    generic.test_read(ds, param, indices, scheduler)


def test_subset(olci_level1):
    ds = Level1(olci_level1)
    generic.test_subset(ds)

@pytest.mark.parametrize('s', [
    slice(None, 300),
    slice(None, None, 10),
    ])
@pytest.mark.parametrize('interp', ['legacy', 'linear', 'atan2'])
def test_preview(olci_level1, param, request, s, interp):
    """
    Generate browses of various products
    """
    plt.figure()
    l1 = Level1(olci_level1, interp_angles=interp)
    eo.init_Rtoa(l1)
    if l1[param].ndim == 2:
        plt.imshow(l1[param][s, s])
    elif l1[param].ndim == 3:
        plt.imshow(l1[param][-1, s, s])
    plt.colorbar()
    savefig(request)


def test_flag(olci_level1):
    """
    Check that flags are properly raised
    """
    l1 = Level1(olci_level1)
    assert (l1.quality_flags > 0).any()
    assert (l1.flags > 0).any()
