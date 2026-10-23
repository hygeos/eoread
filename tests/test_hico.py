#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

import pytest
from core.env import getvar

from eoread.hico import Level1_HICO, FlagsReader_HICO, HICO_BANDS, get_sample
from tests import generic


def _hico_sample() -> Path | None:
    p = getvar("LEVEL1_HICO", default="")
    if not p or not Path(p).exists():
        return None
    return Path(p)


@pytest.fixture(scope="module")
def level1_hico() -> Path:
    sample = _hico_sample()
    if sample is None:
        pytest.skip("LEVEL1_HICO is not defined or does not exist")
    return sample


@pytest.fixture(params=[500, (200, 300)])
def chunks(request):
    return request.param


@pytest.fixture
def hico_product(level1_hico, chunks):
    return Level1_HICO(level1_hico, chunks=chunks)


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_hico, chunks):
    Level1_HICO(level1_hico, chunks=chunks)


def test_main(hico_product):
    generic.Test.main(hico_product, angle_data=True)


def test_bands(hico_product):
    # 128 bands, identified by their nominal central wavelengths (as strings)
    assert len(hico_product.bands) == len(HICO_BANDS) == 128
    assert list(hico_product.bands.values) == [str(b) for b in HICO_BANDS]

    # central wavelengths from the product, in nm
    cwav = hico_product["cwav"].values
    assert cwav.dtype.kind == "f"
    assert (cwav > 300).all() and (cwav < 1100).all()
    # first and last band central wavelengths, in nm
    assert abs(hico_product.cwav[0].values - 352.5) < 0.5
    assert abs(hico_product.cwav[-1].values - 1080.0) < 0.5


def test_units(hico_product):
    assert hico_product["Ltoa"].attrs.get("units") == "W/m^2/micrometer/sr"


def test_attributes(hico_product):
    assert hico_product.attrs["sensor"] == "HICO"
    assert hico_product.attrs["platform"] == "ISS"
    assert hico_product.attrs["_srf_getter"] == "eotools.srf.get_SRF_NASA"
    assert hico_product.attrs["_srf_getter_arg"] == "iss_hico"


def test_flagreader(hico_product):
    # HICO L1B products carry no per-pixel quality flags: all flags are False
    generic.Test.flagreader(hico_product)


def test_get_sample():
    sample = _hico_sample()
    if sample is None:
        pytest.skip("LEVEL1_HICO is not defined or does not exist")
    assert get_sample() == sample
