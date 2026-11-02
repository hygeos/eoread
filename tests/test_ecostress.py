#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the ECOSTRESS readers (Phase 2).

Samples (see tests/README.md):
- ECOSTRESS_L1C: ISS-ECOSTRESS L1CG gridded radiance granule (ECOv002,
  70 m grid of 7074 x 8871 pixels, night pass over the western US).
- ECOSTRESS_L1B: v001 TRISHNA maquette L1B_RAD granule over India; the
  paired L1B_GEO file sits next to it and is auto-discovered by the reader.
- ECOSTRESS_L2_V001: the matching v001 L2_LSTE granule (same GEO pair).
- ECOSTRESS_L2_V002: an ECOv002_L2_LSTE granule (~25% water, coastal
  scene) with its paired ECOv002_L1B_GEO file in the same directory.

No v1 reference exists for the L1B/L2 products (the v1 L2 entry point targeted
a dead HDF-EOS5 path), so their verification is schema conformance plus
physical sanity on real data. The L1B/L2 samples live on a network mount, so
computing tests keep their read windows small and `generic.plot` is not
exercised for them (it materializes full grids).
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.geo.naming import names

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.flags import FlagsInit, GenericFlags
from eoread.reader import ProductError, UnsupportedProductType
from eoread.readers.ecostress import (
    Level1B_ECOSTRESS,
    Level1C_ECOSTRESS,
    Level2_ECOSTRESS,
)
from eoread.readers.msi import Level1C_MSI


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles, so reader datasets are only freed
    # by the cyclic GC. Force a collection after every test to keep the
    # session memory bounded (see test_msi.py).
    yield
    gc.collect()


@pytest.fixture(scope='session')
def level1_ecostress():
    return get_sample(Level1C_ECOSTRESS)


@pytest.fixture(scope='session')
def level1b_ecostress():
    return get_sample(Level1B_ECOSTRESS)


@pytest.fixture(scope='session')
def level2_ecostress():
    return get_sample(Level2_ECOSTRESS, 'ECOSTRESS_L2_V001')


@pytest.fixture(scope='session')
def level2_ecostress_v002():
    path = get_sample(Level2_ECOSTRESS, 'ECOSTRESS_L2_V002')
    # The v002 LSTE product carries no per-pixel lat/lon: it is only readable
    # with its paired L1B_GEO file next to it. Skip cleanly when the staged
    # sample lacks that companion rather than erroring in every dependent test.
    geo = path.with_name(path.name.replace('L2_LSTE', 'L1B_GEO'))
    if not geo.exists():
        pytest.skip(f'missing paired geolocation file {geo.name}')
    return path


@pytest.fixture(scope='session')
def level1_msi():
    return get_sample(Level1C_MSI)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


@pytest.fixture
def l1(level1_ecostress, chunks):
    return Level1C_ECOSTRESS(level1_ecostress, chunks=chunks)


@pytest.fixture
def l1b(level1b_ecostress, chunks):
    return Level1B_ECOSTRESS(level1b_ecostress, chunks=chunks)


@pytest.fixture
def l2(level2_ecostress, chunks):
    return Level2_ECOSTRESS(level2_ecostress, chunks=chunks)


@pytest.fixture
def l2_v002(level2_ecostress_v002, chunks):
    return Level2_ECOSTRESS(level2_ecostress_v002, chunks=chunks)


def _window(ds, rows=slice(200, 400), cols=slice(500, 700)):
    """Small compute window: the samples live on a network mount."""
    return ds.isel({str(names.rows): rows, str(names.columns): cols})


def _exit_probe(cls, sample, tmp_path):
    """Build one reader in a subprocess held to exit; assert clean teardown."""
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        f"from eoread.readers.ecostress import {cls.__name__}\n"
        f"ds = {cls.__name__}({str(sample)!r})\n"
        f"assert ds is not None\n"
        f"print('READER_BUILT')\n",
    )
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'READER_BUILT' in proc.stdout
    assert 'excepthook' not in proc.stderr, \
        'teardown noise at exit: the HDF5 file managers were not closed cleanly'


# ---------------------------------------------------------------------------
# L1C (HDF-EOS5 gridded radiance)
# ---------------------------------------------------------------------------

def test_l1c_instantiation(level1_ecostress, chunks):
    Level1C_ECOSTRESS(level1_ecostress, chunks=chunks)


def test_l1c_main(l1):
    generic.Test.main(l1, angle_data=False)


def test_l1c_conformance(l1):
    generic.Test.v2_conformance(l1, primary='Ltoa')


def test_l1c_latlon(l1):
    generic.Test.latlon(l1, flat_proj=True)


def test_l1c_lazy_load(l1):
    generic.Test.lazy_load(l1)


def test_l1c_lazy_payload(level1_ecostress):
    """Construction must not materialize full-resolution rasters.

    Regression guard for an eager read: every array value in the payload
    graph must be strictly smaller than one full grid (the only ndarrays the
    reader embeds are the 5-element calibration tables).
    """
    ds = Level1C_ECOSTRESS(level1_ecostress)
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for value in ds[str(names.ltoa)].data.__dask_graph__().values():
        if isinstance(value, np.ndarray):
            assert value.size < ny * nx, \
                'full-resolution array materialized at construction time'


def test_l1c_exit_clean_stderr(level1_ecostress, tmp_path):
    """A reader held until process exit must not emit teardown noise.

    The h5netcdf file stays open for lazy reads and is closed by the
    interpreter at finalization; that path must come back clean (same guard
    as test_msi.py's rasterio regression).
    """
    _exit_probe(Level1C_ECOSTRESS, level1_ecostress, tmp_path)


def test_l1c_flagreader(l1):
    generic.Test.flagreader(l1)


def test_l1c_subset(level1_ecostress, chunks):
    ds = Level1C_ECOSTRESS(level1_ecostress, chunks=chunks, metadata_template=[])
    generic.Test.subset(ds)


def test_l1c_plot(request, level1_ecostress):
    ds = Level1C_ECOSTRESS(level1_ecostress)
    generic.plot(request, ds, 'B4', poi={'x': 1000, 'y': 3000})


def test_l1c_time(level1_ecostress, chunks):
    params = {'path': level1_ecostress, 'chunks': chunks}
    generic.Test.execution_time(Level1C_ECOSTRESS, params)


def test_l1c_v1_compat(level1_ecostress):
    ds = Level1C_ECOSTRESS(level1_ecostress)
    old = v1(ds)
    # No variable renames for ECOSTRESS: the payload keeps its canonical name
    assert 'Ltoa' in old
    assert 'schema_version' not in old.attrs


def test_l1c_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        Level1C_ECOSTRESS(tmp_path / 'missing.h5')


def test_l1c_foreign_product(level1_msi):
    """Pointing the reader at a non-HDF-EOS5 product must raise ProductError."""
    with pytest.raises(ProductError):
        Level1C_ECOSTRESS(level1_msi)


# ---------------------------------------------------------------------------
# L1B (v001 L1B_RAD)
# ---------------------------------------------------------------------------

def test_l1b_instantiation(l1b):
    assert l1b is not None


def test_l1b_main(l1b):
    generic.Test.main(l1b, angle_data=True)


def test_l1b_conformance(l1b):
    generic.Test.v2_conformance(l1b, primary=str(names.ltoa))


def test_l1b_latlon(l1b):
    # The swath is a tilted parallelogram: rows are not lines of constant
    # latitude, so the flat-projection invariants do not apply.
    generic.Test.latlon(l1b, flat_proj=False)


def test_l1b_lazy_load(l1b):
    generic.Test.lazy_load(l1b)


def test_l1b_lazy_payload(level1b_ecostress):
    """Construction must not materialize full-resolution rasters.

    Regression guard for an eager read: every array value in the payload
    graph must be strictly smaller than one full grid (the only ndarrays the
    reader embeds are the 5-element wavelength table).
    """
    ds = Level1B_ECOSTRESS(level1b_ecostress)
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for value in ds[str(names.ltoa)].data.__dask_graph__().values():
        if isinstance(value, np.ndarray):
            assert value.size < ny * nx, \
                'full-resolution array materialized at construction time'


def test_l1b_exit_clean_stderr(level1b_ecostress, tmp_path):
    _exit_probe(Level1B_ECOSTRESS, level1b_ecostress, tmp_path)


def test_l1b_flagreader(l1b):
    # Subset to bound the network read; the flag machinery is chunk-agnostic.
    sub = l1b.isel({str(names.rows): slice(100, 300),
                    str(names.columns): slice(200, 400)})
    generic.Test.flagreader(sub)


def test_l1b_subset(level1b_ecostress, chunks):
    ds = Level1B_ECOSTRESS(level1b_ecostress, chunks=chunks, metadata_template=[])
    generic.Test.subset(ds)


def test_l1b_time(level1b_ecostress, chunks):
    params = {'path': level1b_ecostress, 'chunks': chunks}
    generic.Test.execution_time(Level1B_ECOSTRESS, params)


def test_l1b_v1_compat(level1b_ecostress):
    ds = Level1B_ECOSTRESS(level1b_ecostress)
    old = v1(ds)
    # No variable renames for ECOSTRESS: the payload keeps its canonical name.
    assert 'Ltoa' in old
    assert 'schema_version' not in old.attrs


def test_l1b_sanity(l1b):
    """Physical sanity on the v001 sample (no v1 reference exists)."""
    bt = _window(l1b)[str(names.bt)].values
    assert (~np.isnan(bt)).any()
    assert 280. <= np.nanmin(bt) and np.nanmax(bt) <= 340., \
        f'BT outside [280, 340] K: [{np.nanmin(bt)}, {np.nanmax(bt)}]'
    lat = _window(l1b)[str(names.lat)].values
    assert 16. < lat.min() and lat.max() < 22., \
        f'latitude outside the India scene: [{lat.min()}, {lat.max()}]'


def test_l1b_foreign_product(level2_ecostress):
    """A LSTE file is not a L1B product: loud raise at open time."""
    with pytest.raises(UnsupportedProductType):
        Level1B_ECOSTRESS(level2_ecostress)


def test_l1b_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        Level1B_ECOSTRESS(tmp_path / 'missing.h5')


# ---------------------------------------------------------------------------
# L2 v001 (ECO2LSTE)
# ---------------------------------------------------------------------------

def test_l2_instantiation(l2):
    assert l2 is not None


def test_l2_main(l2):
    # Bandless payload: LST plays the role of the radiance variable.
    generic.Test.main(l2, angle_data=True, primary='LST')


def test_l2_conformance(l2):
    generic.Test.v2_conformance(l2, primary='LST')


def test_l2_latlon(l2):
    generic.Test.latlon(l2, flat_proj=False)


def test_l2_lazy_load(l2):
    generic.Test.lazy_load(l2)


def test_l2_lazy_payload(level2_ecostress):
    """Construction must not materialize full-resolution rasters."""
    ds = Level2_ECOSTRESS(level2_ecostress)
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for value in ds['LST'].data.__dask_graph__().values():
        if isinstance(value, np.ndarray):
            assert value.size < ny * nx, \
                'full-resolution array materialized at construction time'


def test_l2_exit_clean_stderr(level2_ecostress, tmp_path):
    _exit_probe(Level2_ECOSTRESS, level2_ecostress, tmp_path)


def test_l2_flagreader(l2):
    sub = l2.isel({str(names.rows): slice(100, 300),
                   str(names.columns): slice(200, 400)})
    generic.Test.flagreader(sub)


def test_l2_subset(level2_ecostress, chunks):
    ds = Level2_ECOSTRESS(level2_ecostress, chunks=chunks, metadata_template=[])
    generic.Test.subset(ds)


def test_l2_time(level2_ecostress, chunks):
    params = {'path': level2_ecostress, 'chunks': chunks}
    generic.Test.execution_time(Level2_ECOSTRESS, params)


def test_l2_v1_compat(level2_ecostress):
    ds = Level2_ECOSTRESS(level2_ecostress)
    old = v1(ds)
    assert 'LST' in old
    assert 'schema_version' not in old.attrs


def test_l2_sanity(l2):
    """Physical sanity on the v001 sample (no v1 reference exists)."""
    w = _window(l2)
    lst = w['LST'].values
    assert (~np.isnan(lst)).any()
    assert 280. <= np.nanmin(lst) and np.nanmax(lst) <= 340., \
        f'LST outside [280, 340] K: [{np.nanmin(lst)}, {np.nanmax(lst)}]'
    emis = w['emis'].sel({str(names.bands): 'B1'}).values
    assert (~np.isnan(emis)).any()
    assert 0. <= np.nanmin(emis) and np.nanmax(emis) <= 1., \
        f'emissivity outside [0, 1]: [{np.nanmin(emis)}, {np.nanmax(emis)}]'


def test_l2_cloud_flag_raises(l2):
    """v001 carries no cloud layer: the CLOUD flag must raise loudly."""
    sub = l2.isel({str(names.rows): slice(100, 300),
                   str(names.columns): slice(200, 400)}).compute()
    with pytest.raises(ValueError):
        Level2_ECOSTRESS.getflag(sub, GenericFlags.CLOUD)


def test_l2_foreign_product():
    """A CLOUD product is not a LSTE product: loud raise at open time."""
    # The v002 LSTE staging ships the CLOUD sibling (the v001 sample does not);
    # this test only needs the file path, not a readable LSTE+GEO pair.
    lste = get_sample(Level2_ECOSTRESS, 'ECOSTRESS_L2_V002')
    cloud = lste.with_name(lste.name.replace('L2_LSTE', 'L2_CLOUD'))
    if not cloud.exists():
        pytest.skip(f'missing CLOUD sibling {cloud.name}')
    with pytest.raises(UnsupportedProductType):
        Level2_ECOSTRESS(cloud)


def test_l2_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        Level2_ECOSTRESS(tmp_path / 'missing.h5')


# ---------------------------------------------------------------------------
# L2 v002 (ECO_L2_LSTE, in-grid cloud mask)
# ---------------------------------------------------------------------------

def test_l2_v002_instantiation(l2_v002):
    assert l2_v002 is not None


def test_l2_v002_main(l2_v002):
    generic.Test.main(l2_v002, angle_data=True, primary='LST')


def test_l2_v002_conformance(l2_v002):
    generic.Test.v2_conformance(l2_v002, primary='LST')


def test_l2_v002_latlon(l2_v002):
    generic.Test.latlon(l2_v002, flat_proj=False)


def test_l2_v002_lazy_load(l2_v002):
    generic.Test.lazy_load(l2_v002)


def test_l2_v002_flagreader(l2_v002):
    # The reader itself is the flag provider, and it must see the v002
    # cloud layer.
    assert l2_v002.attrs['_flag_reader'] == (
        'eoread.readers.ecostress.Level2_ECOSTRESS')
    sub = l2_v002.isel({str(names.rows): slice(100, 300),
                        str(names.columns): slice(200, 400)})
    generic.Test.flagreader(sub)


def test_l2_v002_cloud_flag(l2_v002):
    """The v002 in-grid cloud mask decodes through the blockwise machinery."""
    sub = l2_v002.isel({str(names.rows): slice(100, 300),
                        str(names.columns): slice(200, 400)})
    init = FlagsInit({GenericFlags.CLOUD: 1}, 'uint8',
                     l2_v002.attrs['_flag_reader'])
    out = init.map_blocks(sub)
    cloud = (out['flags'] & 1).values.astype(bool)
    assert (cloud == (sub['cloud'] == 1).values).all()


def test_l2_v002_sanity(l2_v002):
    w = _window(l2_v002)
    lst = w['LST'].values
    assert (~np.isnan(lst)).any()
    # The window sits over open water (Seine estuary, February pass): LST
    # tracks the ~275 K winter sea surface there, far below land values.
    assert 260. <= np.nanmin(lst) and np.nanmax(lst) <= 350., \
        f'LST outside [260, 350] K: [{np.nanmin(lst)}, {np.nanmax(lst)}]'
    frac = float(l2_v002['water'].mean())
    assert 0.2 < frac < 0.3, \
        f'expected ~25% water on the coastal scene, got {frac:.1%}'


def test_l2_v002_time(level2_ecostress_v002):
    generic.Test.execution_time(Level2_ECOSTRESS, {'path': level2_ecostress_v002})


# ---------------------------------------------------------------------------
# Cross-reader guards
# ---------------------------------------------------------------------------

def test_l1c_rejects_plain_product(level2_ecostress):
    """The HDF-EOS5 L1C reader must reject the plain-HDF5 products."""
    with pytest.raises(UnsupportedProductType):
        Level1C_ECOSTRESS(level2_ecostress)
