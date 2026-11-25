#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the Sentinel-2 MSI readers (Phase 1).

Samples (see tests/README.md): MSI_L1C is an S2A L1C baseline-5 product;
MSI_L2A is an S2B L2A baseline-5.11 product (no B10, B8 present in R10m).
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hygeos_core.geo.naming import names

from eoread.flags import GenericFlags

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.readers.msi import Level1C_MSI, Level2A_MSI


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles, so reader datasets are only freed
    # by the cyclic GC. Force a collection after every test to keep the
    # session memory bounded (each MSI instance still holds thousands of graph
    # nodes; letting dozens accumulate across a long session bloats RSS).
    yield
    gc.collect()


@pytest.fixture(scope='session')
def level1_msi():
    return get_sample(Level1C_MSI)


@pytest.fixture(scope='session')
def level2_msi():
    return get_sample(Level2A_MSI)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


@pytest.fixture(params=[10, 20, 60])
def resolution(request):
    return request.param


@pytest.fixture
def l1(level1_msi, chunks, resolution):
    return Level1C_MSI(level1_msi, chunks=chunks, resolution=resolution)


@pytest.fixture
def l2(level2_msi, chunks, resolution):
    return Level2A_MSI(level2_msi, chunks=chunks, resolution=resolution)


#: Standard Sentinel-2 band -> native resolution (m) table.
_BAND_RES = {'B1': 60, 'B2': 10, 'B3': 10, 'B4': 10, 'B5': 20, 'B6': 20, 'B7': 20,
             'B8': 10, 'B8A': 20, 'B8AO': 60, 'B9': 60, 'B10': 60, 'B11': 20, 'B12': 20}


def _meta_bands(ds):
    """Product-declared band labels, in metadata (ascending-cwav) order."""
    specs = ds.attrs['metadata']['General_Info']['Product_Image_Characteristics'] \
        ['Spectral_Information_List']['Spectral_Information']
    return [s['attributes']['physicalBand'] for s in specs]


def _assert_native_layout(ds, family, exact_meta=True):
    """Structural checks of the native (resolution=None) payload layout.

    With `exact_meta` (L1C: every declared band must have a file) the groups
    cover exactly the product band table; otherwise (L2A: bands may be absent
    from the product, e.g. B10) they must stay within it.
    """
    # No uniform-layout variables; the per-group payloads carry the spectral data.
    assert str(names.bands) not in ds.dims and str(names.cwav) not in ds
    meta = _meta_bands(ds)
    seen = []
    for res in (10, 20, 60):
        pv, bdim, cv = f'{family}_{res}m', f'bands_{res}m', f'cwav_{res}m'
        if bdim not in ds:
            assert pv not in ds and cv not in ds
            continue
        # Payload stack on its own (bands, y, x) grid with pixel-index coords.
        var = ds[pv]
        assert var.dims == (bdim, f'y_{res}m', f'x_{res}m')
        assert list(ds[f'y_{res}m'].values) == list(range(var.sizes[f'y_{res}m']))
        assert list(ds[f'x_{res}m'].values) == list(range(var.sizes[f'x_{res}m']))
        # Scientific labels, all at this resolution, in metadata (ascending-cwav)
        # order.
        labels = list(ds[bdim].values)
        seen.extend(labels)
        assert all(_BAND_RES[b] == res for b in labels), \
            f'{bdim}: {labels} not all at {res} m'
        assert labels == [b for b in meta if b in set(labels)], \
            f'{bdim}: {labels} not in metadata order'
        # Per-group cwav table: finite, positive, strictly ascending.
        cw = ds[cv]
        assert cw.dims == (bdim,) and cw.attrs.get('unit') == 'nm'
        vals = cw.values
        assert np.all(np.isfinite(vals)) and np.all(vals > 0)
        assert np.all(np.diff(vals) > 0), f'{cv}: not ascending'
    # The groups never invent bands; L1C drops none.
    assert set(seen) <= set(meta)
    if exact_meta:
        assert sorted(seen) == sorted(meta)
    # Geometry on the canonical reference grid at the coarsest resolution.
    assert ds.attrs[str(names.resolution)] == 60
    for v in (str(names.lat), str(names.lon), str(names.sza), str(names.saa),
              str(names.vza), str(names.vaa)):
        assert ds[v].dims == (str(names.rows), str(names.columns)), v


def _native_parity(reader_cls, path):
    """Native band values must be bit-identical to a uniform read at the same resolution."""
    ds = reader_cls(path, resolution=None)
    family = 'Rtoa'
    for res in (10, 20, 60):
        bdim = f'bands_{res}m'
        if bdim not in ds:
            continue
        uni = reader_cls(path, resolution=res)
        try:
            for label in ds[bdim].values:
                a = ds[f'{family}_{res}m'].sel({bdim: label}).isel(
                    **{f'y_{res}m': slice(1000, 1200), f'x_{res}m': slice(1000, 1200)}).compute()
                b = uni[family].sel(bands=label).isel(
                    y=slice(1000, 1200), x=slice(1000, 1200)).compute()
                assert np.array_equal(a, b), \
                    f'{label}: native and res-{res} values differ'
        finally:
            del uni


################################################################################
# Level-1C
################################################################################

def test_l1_instantiation(level1_msi, chunks, resolution):
    Level1C_MSI(level1_msi, chunks=chunks, resolution=resolution)


def test_l1_main(l1):
    generic.Test.main(l1, angle_data=True)


def test_l1_conformance(l1):
    generic.Test.v2_conformance(l1, primary='Rtoa')


def test_l1_latlon(l1):
    generic.Test.latlon(l1, flat_proj=False)


def test_l1_lazy_load(l1):
    generic.Test.lazy_load(l1)


def test_l1_lazy_bands_r10(level1_msi):
    """Construction must not materialize full-resolution band rasters.

    Regression guard for the eager band open: _read_bands used to open each
    band file unchunked and let dask.array.from_array copy every calibrated
    float32 raster into the graph (~2.7 GB held at construction for a res-10
    L1C). Bands must be opened already chunked, so every array value in the
    payload graph is a per-window read strictly smaller than one full band.

    Named with the r10 suffix so the staged runner (-k "l1 and 10") runs it
    in the res-10 stage, where a full-resolution regression would be largest.
    """
    ds = Level1C_MSI(level1_msi, resolution=10)
    ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
    for value in ds[str(names.rtoa)].data.__dask_graph__().values():
        if isinstance(value, np.ndarray):
            assert value.size < ny * nx, \
                'full-resolution band array materialized at construction time'


def test_l1_exit_clean_stderr_r10(level1_msi, tmp_path):
    """A reader held until process exit must not emit teardown noise.

    Regression guard for the atexit file-manager close (_install_exit_cleanup
    in msi.py): rioxarray leaves one rasterio dataset open per band for lazy
    windowed reads, and if such a dataset is still open when the interpreter
    finalizes, CPython >= 3.12 prints a nested 'Error in sys.excepthook' pair
    with an empty traceback for each of them (RC stays 0; results are fine).
    The reader must be built in a child process — the noise only appears at
    finalization — and that process's stderr must come back clean.

    Named with the r10 suffix so the staged runner (-k "l1 and 10") runs it
    in the res-10 stage, where the number of open band files is largest.
    """
    script = tmp_path / 'exit_probe.py'
    script.write_text(
        "from eoread.readers.msi import Level1C_MSI\n"
        f"ds = Level1C_MSI({str(level1_msi)!r}, resolution=10)\n"
        "assert ds is not None\n"
        "print('READER_BUILT')\n",
    )
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'READER_BUILT' in proc.stdout
    assert 'excepthook' not in proc.stderr, \
        'teardown noise at exit: band file managers were not closed by atexit'


def test_l1_flagreader(l1):
    generic.Test.flagreader(l1)


def test_l1_subset(level1_msi, chunks):
    ds = Level1C_MSI(level1_msi, chunks=chunks, metadata_template=[])
    generic.Test.subset(ds)


def test_l1_plot(request, level1_msi):
    ds = Level1C_MSI(level1_msi, resolution=60)
    generic.plot(request, ds, 'B4', poi={'x': 1000, 'y': 1000})


def test_l1_time(level1_msi, chunks):
    params = {'path': level1_msi, 'chunks': chunks}
    generic.Test.execution_time(Level1C_MSI, params)


def test_l1_v1_compat(level1_msi):
    ds = Level1C_MSI(level1_msi, resolution=60)
    old = v1(ds)
    assert 'Rtoa' in old
    assert 'schema_version' not in old.attrs


def test_l1_resolution_none(level1_msi):
    """Native layout (resolution=None): per-resolution stacks on their own grids."""
    ds = Level1C_MSI(level1_msi, resolution=None)
    # No uniform-layout payload; the finest stack is the declared primary.
    assert 'Rtoa' not in ds and 'Rtoa_10m' in ds
    _assert_native_layout(ds, 'Rtoa')


def test_l1_native_parity(level1_msi):
    """Native bands must be bit-identical to a uniform read at the same resolution."""
    _native_parity(Level1C_MSI, level1_msi)


@pytest.fixture(scope='module')
def l1_angle_all(level1_msi):
    return Level1C_MSI(level1_msi, resolution=60, angle_band='all')


@pytest.fixture(scope='module')
def l1_angle_b3(level1_msi):
    return Level1C_MSI(level1_msi, resolution=60, angle_band='B3')


_ROI = {'y': slice(500, 700), 'x': slice(500, 700)}


def test_l1_angle_default(level1_msi):
    """Default (angle_band=None): viewing geometry of B1, 2D, as bandId 0."""
    ds = Level1C_MSI(level1_msi, resolution=60)
    ds0 = Level1C_MSI(level1_msi, resolution=60, angle_band=0)
    for v in (names.sza, names.saa, names.vza, names.vaa):
        assert ds[str(v)].dims == (str(names.rows), str(names.columns))
    for v in (names.vza, names.vaa):
        assert np.array_equal(ds[str(v)].isel(_ROI), ds0[str(v)].isel(_ROI), equal_nan=True)


def test_l1_angle_single_band(level1_msi, l1_angle_b3):
    """A single band, by name or by bandId: same 2D viewing geometry."""
    ds_id = Level1C_MSI(level1_msi, resolution=60, angle_band=2)
    ds_def = Level1C_MSI(level1_msi, resolution=60)
    for v in (names.vza, names.vaa):
        assert l1_angle_b3[str(v)].dims == (str(names.rows), str(names.columns))
        a = l1_angle_b3[str(v)].isel(_ROI).values
        assert np.array_equal(a, ds_id[str(v)].isel(_ROI).values, equal_nan=True)
        # differs from the default band (B1)
        assert not np.array_equal(a, ds_def[str(v)].isel(_ROI).values, equal_nan=True)


def test_l1_angle_all(l1_angle_all, l1_angle_b3):
    """angle_band='all': (bands, y, x) viewing angles, 2D solar angles."""
    ds = l1_angle_all
    bands, rows, cols = str(names.bands), str(names.rows), str(names.columns)
    for v in (names.vza, names.vaa):
        assert ds[str(v)].dims == (bands, rows, cols)
        assert ds[str(v)].dtype == np.float32
        # each band matches the single-band read
        assert np.array_equal(ds[str(v)].sel({bands: 'B3'}).isel(_ROI).values,
                              l1_angle_b3[str(v)].isel(_ROI).values, equal_nan=True)
    for v in (names.sza, names.saa):
        assert ds[str(v)].dims == (rows, cols)
    vza = ds[str(names.vza)].isel(_ROI).values
    assert np.isfinite(vza).any()
    # the viewing geometry differs between bands
    assert not np.array_equal(vza[0], vza[1], equal_nan=True)


def test_l1_angle_all_conformance(l1_angle_all):
    generic.Test.v2_conformance(l1_angle_all, primary='Rtoa')


def test_l1_angle_all_flags(l1_angle_all, level1_msi):
    """With per-band geometry, L1_INVALID stays 2D: invalid where vza is NaN in any band."""
    rows, cols = str(names.rows), str(names.columns)
    ds_def = Level1C_MSI(level1_msi, resolution=60)
    f_all = Level1C_MSI.getflag(l1_angle_all, GenericFlags.L1_INVALID).compute()
    f_def = Level1C_MSI.getflag(ds_def, GenericFlags.L1_INVALID).compute()
    assert f_all.dims == f_def.dims == (rows, cols)
    # pixels invalid in band B1 are also invalid with all bands
    assert bool((f_all | ~f_def).all())
    generic.Test.flagreader(l1_angle_all.isel({rows: slice(0, 200), cols: slice(0, 200)}))


def test_l1_angle_invalid(level1_msi):
    for angle_band in (99, 'B99', 1.5, True):
        with pytest.raises(ValueError):
            Level1C_MSI(level1_msi, resolution=60, angle_band=angle_band)
    with pytest.raises(ValueError):
        Level1C_MSI(level1_msi, resolution=None, angle_band='all')


def test_l2_angle_all(level2_msi):
    """L2A: one viewing geometry per band actually read (no B10)."""
    ds = Level2A_MSI(level2_msi, resolution=60, angle_band='all')
    assert ds[str(names.vza)].dims == (str(names.bands), str(names.rows), str(names.columns))
    assert 'B10' not in ds[str(names.bands)].values


def test_l1_native_v1_compat(level1_msi):
    """Native stacks keep their Rtoa names (no rename)."""
    ds = Level1C_MSI(level1_msi, resolution=None)
    old = v1(ds)
    assert 'Rtoa_10m' in old
    assert 'schema_version' not in old.attrs


################################################################################
# Level-2A
################################################################################

def test_l2_instantiation(level2_msi, chunks, resolution):
    Level2A_MSI(level2_msi, chunks=chunks, resolution=resolution)


def test_l2_main(l2):
    generic.Test.main(l2, angle_data=True)


def test_l2_conformance(l2):
    generic.Test.v2_conformance(l2, primary='Rtoa')


def test_l2_latlon(l2):
    generic.Test.latlon(l2, flat_proj=False)


def test_l2_lazy_load(l2):
    generic.Test.lazy_load(l2)


def test_l2_flagreader(l2):
    generic.Test.flagreader(l2)


def test_l2_bands(l2):
    # Baseline-5 L2A: B10 is absent from the product and B8 is kept (v1 dropped it).
    assert list(l2.coords['bands'].values) == \
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']


def test_l2_resolution_none(level2_msi):
    """Native layout for L2A: the sample's exact per-resolution grouping."""
    ds = Level2A_MSI(level2_msi, resolution=None)
    # No uniform-layout payload; the finest stack is the declared primary.
    assert 'Rtoa' not in ds and 'Rtoa_10m' in ds
    _assert_native_layout(ds, 'Rtoa', exact_meta=False)
    # Baseline-5 L2A sample (test_l2_bands): B10/B8AO absent; exact groups from
    # the standard Sentinel-2 band/resolution table.
    assert list(ds['bands_10m'].values) == ['B2', 'B3', 'B4', 'B8']
    assert list(ds['bands_20m'].values) == ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
    assert list(ds['bands_60m'].values) == ['B1', 'B9']


def test_l2_native_parity(level2_msi):
    """Native bands must be bit-identical to a uniform read at the same resolution."""
    _native_parity(Level2A_MSI, level2_msi)


def test_l2_native_v1_compat(level2_msi):
    """L2A native stacks keep their Rtoa names; schema_version is stripped."""
    ds = Level2A_MSI(level2_msi, resolution=None)
    old = v1(ds)
    assert 'Rtoa_10m' in old and 'Ltoa_10m' not in old
    assert 'schema_version' not in old.attrs
