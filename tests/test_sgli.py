#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from eoread.sgli import Level1_SGLI, calc_central_wavelength, get_sample

from . import generic
from .generic import indices, param  # noqa

sgli_filename = get_sample()

def test_instantiate():
    Level1_SGLI(sgli_filename)


def test_central_wavelength():
    l1 = Level1_SGLI(sgli_filename)
    _, central_wav = calc_central_wavelength()

    np.testing.assert_allclose(l1.wav, central_wav)


def test_main():
    ds = Level1_SGLI(sgli_filename)

    generic.test_main(ds)


def test_read(param, indices):
    ds = Level1_SGLI(sgli_filename)
    generic.test_read(ds, param, indices, scheduler='sync')


def test_subset():
    ds = Level1_SGLI(sgli_filename)
    generic.test_subset(ds)
