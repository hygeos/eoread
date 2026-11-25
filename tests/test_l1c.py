#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the L1C generation dispatcher (`eoread.transforms.l1c.makeL1C`).

The sensor-specific l2gen wrappers are monkeypatched, so no SeaDAS/OCSSW
installation (or docker) is needed: the tests pin the filename -> sensor
dispatch order and the argument pass-through. In particular, the
`SNPP_VIIRS` and `SEAHAWK1_HAWKEYE` prefixes must be matched before the generic
`S` (SeaWiFS) prefix they start with.
"""

from pathlib import Path

import pytest

from eoread.transforms import l1c as l1c_mod


@pytest.fixture
def dispatch(monkeypatch):
    """Monkeypatch the four per-sensor wrappers to record their calls."""
    calls = []

    def make_fake(name):
        def fake(l1a, dirname=None, method='shell', ocssw_version=None,
                 ocssw_tags=None, **kwargs):
            calls.append({'fn': name, 'l1a': l1a, 'dirname': dirname,
                          'method': method, 'ocssw_version': ocssw_version,
                          'ocssw_tags': ocssw_tags, 'kwargs': kwargs})
            return Path('/fake') / f'{name}_L1C.nc'
        return fake

    for fn in ('makeL1C_MODIS', 'makeL1C_VIIRS', 'makeL1C_HAWKEYE',
               'makeL1C_SeaWIFS'):
        monkeypatch.setattr(l1c_mod, fn, make_fake(fn))

    return calls


@pytest.mark.parametrize('name,fn', [
    ('A2023001000000.L1A_LAC', 'makeL1C_MODIS'),
    ('V2024010v01.L1A_LAC', 'makeL1C_VIIRS'),
    # starts with 'S' but must dispatch to VIIRS, not SeaWiFS
    ('SNPP_VIIRS_A2000041.L1A_LAC', 'makeL1C_VIIRS'),
    ('JPSS1_VIIRS_A2024010.L1A_LAC', 'makeL1C_VIIRS'),
    ('JPSS2_VIIRS.20240724T124200.L1A.nc', 'makeL1C_VIIRS'),
    # starts with 'S' but must dispatch to HAWKEYE, not SeaWiFS
    ('SEAHAWK1_HAWKEYE_A2000041.L1A_GAC', 'makeL1C_HAWKEYE'),
    ('S20000417053000.L1A_GAC.bz2', 'makeL1C_SeaWIFS'),
])
def test_dispatch(dispatch, name, fn):
    l1c_mod.makeL1C(name)
    assert len(dispatch) == 1
    assert dispatch[0]['fn'] == fn
    # the input is coerced to a Path and passed through as-is
    assert dispatch[0]['l1a'] == Path(name)
    # ocssw_version/ocssw_tags default to None but are always forwarded
    assert dispatch[0]['ocssw_version'] is None
    assert dispatch[0]['ocssw_tags'] is None


def test_dispatch_args_passthrough(dispatch, tmp_path):
    l1c_mod.makeL1C('A2023001000000.L1A_LAC', dirname=tmp_path,
                    method='docker', gvis=0)
    assert dispatch[0]['dirname'] == tmp_path
    assert dispatch[0]['method'] == 'docker'
    assert dispatch[0]['kwargs'] == {'gvis': 0}


def test_dispatch_ocssw_params_passthrough(dispatch):
    l1c_mod.makeL1C('A2023001000000.L1A_LAC', ocssw_version='V2025.5',
                    ocssw_tags=['latest'])
    assert dispatch[0]['ocssw_version'] == 'V2025.5'
    assert dispatch[0]['ocssw_tags'] == ['latest']


def test_dispatch_unknown_prefix(dispatch):
    with pytest.raises(ValueError, match='Invalid sensor'):
        l1c_mod.makeL1C('X2023001.L1A_LAC')
    assert dispatch == []
