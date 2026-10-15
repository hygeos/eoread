#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the ECOSTRESS readers (Phase 2).

Samples (see tests/README.md):
- LEVEL1_ECOSTRESS: ISS-ECOSTRESS L1CG gridded radiance granule (ECOv002,
  70 m grid of 7074 x 8871 pixels, night pass over the western US).
- LEVEL1B_ECOSTRESS: v001 TRISHNA maquette L1B_RAD granule over India; the
  paired L1B_GEO file sits next to it and is auto-discovered by the reader.
- LEVEL2_ECOSTRESS: the matching v001 L2_LSTE granule (same GEO pair).
- LEVEL2_ECOSTRESS_V002: an ECOv002_L2_LSTE granule (~25% water, coastal
  scene) with its paired ECOv002_L1B_GEO file in the same directory.

No legacy parity reference exists for the L1B/L2 products (the legacy L2
entry point targets a dead HDF-EOS5 path), so their verification is schema
conformance plus physical sanity on real data; the L1C reader is pinned
against the legacy implementation instead. The L1B/L2 samples live on a
network mount, so computing tests keep their read windows small and
`generic.plot` is not exercised for them (it materializes full grids).
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.geo.naming import names

from . import generic
from .conftest import require_sample

from eoread.compat import v1
from eoread.flags import FlagsInit, GenericFlags
from eoread.reader import ProductError, UnsupportedProductType
from eoread.readers.ecostress import (
    FlagsReader_Level2,
    Level1B_Ecostress,
    Level1C_Ecostress,
    Level2_Ecostress,
)


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles, so reader datasets are only freed
    # by the cyclic GC. Force a collection after every test to keep the
    # session memory bounded (see test_msi.py).
    yield
    gc.collect()


@pytest.fixture(scope='session')
def level1_ecostress():
    return require_sample('LEVEL1_ECOSTRESS')


@pytest.fixture(scope='session')
def level1b_ecostress():
    return require_sample('LEVEL1B_ECOSTRESS')


@pytest.fixture(scope='session')
def level2_ecostress():
    return require_sample('LEVEL2_ECOSTRESS')


@pytest.fixture(scope='session')
def level2_ecostress_v002():
    return require_sample('LEVEL2_ECOSTRESS_V002')


@pytest.fixture(scope='session')
def level1_msi():
    return require_sample('LEVEL1_MSI')


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


@pytest.fixture
def l1(level1_ecostress, chunks):
    return Level1C_Ecostress(level1_ecostress, chunks=chunks).ds


@pytest.fixture
def l1b(level1b_ecostress, chunks):
    return Level1B_Ecostress(level1b_ecostress, chunks=chunks).ds


@pytest.fixture
def l2(level2_ecostress, chunks):
    return Level2_Ecostress(level2_ecostress, chunks=chunks).ds


@pytest.fixture
def l2_v002(level2_ecostress_v002, chunks):
    return Level2_Ecostress(level2_ecostress_v002, chunks=chunks).ds


def _window(ds, rows=slice(200, 400), cols=slice(500, 700)):
    """Small compute window: the samples live on a network mount."""
    return ds.isel({str(names.rows): rows, str(names.columns): cols})


def _exit_probe(cls, env_var, tmp_path):
    """Build one reader in a subprocess held to exit; assert clean teardown."""
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        f"import os\n"
        f"from eoread.readers.ecostress import {cls.__name__}\n"
        f"reader = {cls.__name__}(os.environ['{env_var}'])\n"
        f"assert reader.ds is not None\n"
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
    Level1C_Ecostress(level1_ecostress, chunks=chunks)


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
    ds = Level1C_Ecostress(level1_ecostress).ds
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
    _exit_probe(Level1C_Ecostress, 'LEVEL1_ECOSTRESS', tmp_path)


def test_l1c_flagreader(l1):
    generic.Test.flagreader(l1)


def test_l1c_subset(level1_ecostress, chunks):
    ds = Level1C_Ecostress(level1_ecostress, chunks=chunks, metadata_template=[]).ds
    generic.Test.subset(ds)


def test_l1c_plot(request, level1_ecostress):
    ds = Level1C_Ecostress(level1_ecostress).ds
    generic.plot(request, ds, '4', poi={'x': 1000, 'y': 3000})


def test_l1c_time(level1_ecostress, chunks):
    params = {'path': level1_ecostress, 'chunks': chunks}
    generic.Test.execution_time(Level1C_Ecostress, params)


def test_l1c_v1_compat(level1_ecostress):
    ds = Level1C_Ecostress(level1_ecostress).ds
    old = v1(ds)
    # No variable renames for ECOSTRESS: the payload keeps its canonical name
    assert 'Ltoa' in old
    assert 'schema_version' not in old.attrs


def test_l1c_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        Level1C_Ecostress(tmp_path / 'missing.h5')


def test_l1c_foreign_product(level1_msi):
    """Pointing the reader at a non-HDF-EOS5 product must raise ProductError."""
    with pytest.raises(ProductError):
        Level1C_Ecostress(level1_msi)


def test_l1c_parity_legacy(level1_ecostress):
    """Values must match the legacy reader modulo the documented breaks.

    v2 is south-up and stores lat/lon as float32, so v2 row i carries legacy
    row ny-1-i, and lat/lon compare against the flipped legacy values cast to
    float32; everything else must be bit-exact.
    """
    from eoread.legacy.ecostress import Level1C_ECOSTRESS

    v2 = Level1C_Ecostress(level1_ecostress, chunks=500).ds
    legacy = Level1C_ECOSTRESS(level1_ecostress, chunks=500, verbose=False)

    ny, nx = v2.sizes[str(names.rows)], v2.sizes[str(names.columns)]
    rows = [0, ny // 2, ny - 1]
    cols = [0, 42, nx - 1]
    flipped = [ny - 1 - r for r in rows]

    # equal_nan: edge rows carry invalid pixels (band 4 nulls), so the windows
    # contain NaNs that must compare equal to each other.
    for var in (str(names.ltoa), str(names.bt), 'data_quality', 'water', 'cloud'):
        a = v2[var].isel({str(names.rows): rows, str(names.columns): cols}).values
        b = legacy[var].isel({str(names.rows): flipped, str(names.columns): cols}).values
        assert np.array_equal(a, b, equal_nan=True), \
            f'{var} does not match the legacy reader'

    for var in (str(names.lat), str(names.lon)):
        a = v2[var].isel({str(names.rows): rows, str(names.columns): cols}).values
        ref = legacy[var].isel({str(names.rows): flipped, str(names.columns): cols}).values
        assert np.array_equal(a, ref.astype('float32')), f'{var} parity'

    # Band table and scalar attributes
    assert list(v2[str(names.bands)].values) == list(legacy[str(names.bands)].values)
    assert np.array_equal(v2[str(names.cwav)].values, legacy[str(names.cwav)].values)
    assert v2.attrs['metadata'] == legacy.attrs['metadata']
    assert v2.attrs['hdfeos_info'] == legacy.attrs['hdfeos_info']
    for key in ('daytime', 'user_guide', str(names.platform), str(names.sensor),
                str(names.product_name), str(names.resolution)):
        assert v2.attrs[key] == legacy.attrs[key]
    assert Path(v2.attrs[str(names.input_directory)]).resolve() == \
        Path(legacy.attrs[str(names.input_directory)]).resolve()


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
    ds = Level1B_Ecostress(level1b_ecostress).ds
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for value in ds[str(names.ltoa)].data.__dask_graph__().values():
        if isinstance(value, np.ndarray):
            assert value.size < ny * nx, \
                'full-resolution array materialized at construction time'


def test_l1b_exit_clean_stderr(level1b_ecostress, tmp_path):
    _exit_probe(Level1B_Ecostress, 'LEVEL1B_ECOSTRESS', tmp_path)


def test_l1b_flagreader(l1b):
    # Subset to bound the network read; the flag machinery is chunk-agnostic.
    sub = l1b.isel({str(names.rows): slice(100, 300),
                    str(names.columns): slice(200, 400)})
    generic.Test.flagreader(sub)


def test_l1b_subset(level1b_ecostress, chunks):
    ds = Level1B_Ecostress(level1b_ecostress, chunks=chunks, metadata_template=[]).ds
    generic.Test.subset(ds)


def test_l1b_time(level1b_ecostress, chunks):
    params = {'path': level1b_ecostress, 'chunks': chunks}
    generic.Test.execution_time(Level1B_Ecostress, params)


def test_l1b_v1_compat(level1b_ecostress):
    ds = Level1B_Ecostress(level1b_ecostress).ds
    old = v1(ds)
    # No variable renames for ECOSTRESS: the payload keeps its canonical name.
    assert 'Ltoa' in old
    assert 'schema_version' not in old.attrs


def test_l1b_sanity(l1b):
    """Physical sanity on the v001 sample (no legacy parity reference exists)."""
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
        Level1B_Ecostress(level2_ecostress)


def test_l1b_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        Level1B_Ecostress(tmp_path / 'missing.h5')


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
    ds = Level2_Ecostress(level2_ecostress).ds
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for value in ds['LST'].data.__dask_graph__().values():
        if isinstance(value, np.ndarray):
            assert value.size < ny * nx, \
                'full-resolution array materialized at construction time'


def test_l2_exit_clean_stderr(level2_ecostress, tmp_path):
    _exit_probe(Level2_Ecostress, 'LEVEL2_ECOSTRESS', tmp_path)


def test_l2_flagreader(l2):
    sub = l2.isel({str(names.rows): slice(100, 300),
                   str(names.columns): slice(200, 400)})
    generic.Test.flagreader(sub)


def test_l2_subset(level2_ecostress, chunks):
    ds = Level2_Ecostress(level2_ecostress, chunks=chunks, metadata_template=[]).ds
    generic.Test.subset(ds)


def test_l2_time(level2_ecostress, chunks):
    params = {'path': level2_ecostress, 'chunks': chunks}
    generic.Test.execution_time(Level2_Ecostress, params)


def test_l2_v1_compat(level2_ecostress):
    ds = Level2_Ecostress(level2_ecostress).ds
    old = v1(ds)
    assert 'LST' in old
    assert 'schema_version' not in old.attrs


def test_l2_sanity(l2):
    """Physical sanity on the v001 sample (no legacy parity reference exists)."""
    w = _window(l2)
    lst = w['LST'].values
    assert (~np.isnan(lst)).any()
    assert 280. <= np.nanmin(lst) and np.nanmax(lst) <= 340., \
        f'LST outside [280, 340] K: [{np.nanmin(lst)}, {np.nanmax(lst)}]'
    emis = w['emis'].sel({str(names.bands): '1'}).values
    assert (~np.isnan(emis)).any()
    assert 0. <= np.nanmin(emis) and np.nanmax(emis) <= 1., \
        f'emissivity outside [0, 1]: [{np.nanmin(emis)}, {np.nanmax(emis)}]'


def test_l2_cloud_flag_raises(l2):
    """v001 carries no cloud layer: the CLOUD flag must raise loudly."""
    sub = l2.isel({str(names.rows): slice(100, 300),
                   str(names.columns): slice(200, 400)}).compute()
    with pytest.raises(ValueError):
        FlagsReader_Level2().getflag(sub, GenericFlags.CLOUD)


def test_l2_foreign_product(level2_ecostress):
    """A CLOUD product is not a LSTE product: loud raise at open time."""
    cloud = level2_ecostress.with_name(
        level2_ecostress.name.replace('L2_LSTE', 'L2_CLOUD'))
    assert cloud.exists(), 'expected the CLOUD sibling of the LSTE sample'
    with pytest.raises(UnsupportedProductType):
        Level2_Ecostress(cloud)


def test_l2_missing_path(tmp_path):
    with pytest.raises(AssertionError):
        Level2_Ecostress(tmp_path / 'missing.h5')


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
    # The per-variant flag reader (with the cloud layer) must be wired in.
    assert l2_v002.attrs['_flag_reader'].endswith('FlagsReader_Level2Cloud')
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
    generic.Test.execution_time(Level2_Ecostress, {'path': level2_ecostress_v002})


# ---------------------------------------------------------------------------
# Cross-reader guards
# ---------------------------------------------------------------------------

def test_l1c_rejects_plain_product(level2_ecostress):
    """The HDF-EOS5 L1C reader must reject the plain-HDF5 products."""
    with pytest.raises(UnsupportedProductType):
        Level1C_Ecostress(level2_ecostress)
