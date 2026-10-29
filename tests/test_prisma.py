#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 conformance tests for the PRISMA L1 STD reader (ASI).

Real-data policy (tests/README.md): each test reads its sample path from the
sample registry via get_sample(); missing sample means a clean skip.

The archive does not hold a Level2C companion for the sample granule, but the
reader requires one for the per-pixel geometric angles; the session fixture
below therefore stages the real L1 next to a synthesized L2C (constant
grids, sza taken from the L1 global sun angle). Dropping a real L2C next to
the staged granule takes over automatically.

Documented deviations from the v1 implementation (see REFACTORING.md):
per-file south-up / west-to-east flip (the sample is stored row 0 = southern
edge, column 0 = eastern edge, so v2 column j carries raw column nx-1-j),
one dask block per radiance cube (legacy used 500xall-bands x 500 blocks and
re-decompressed the whole cube on every read), the added shortname/crs
attributes, and `ProductError` instead of `AssertionError` when the L2C
companion is missing.

PRISMA emits sza/vza/raa (the L2C has no absolute azimuths), so
`generic.Test.main` runs with angle_data=False and a registry-range check
replaces `generic.Test.angles`.
"""

import gc
from pathlib import Path

import h5py
import numpy as np
import pytest
from core.geo.naming import names

from . import generic
from .conftest import get_sample

from eoread.compat import v1
from eoread.flags import FlagsInit, GenericFlags
from eoread.reader import ProductError, UnsupportedProductType
from eoread.readers.prisma import Level1_PRISMA


def _check_angles(ds):
    """Naming-registry range check on a decimated sample (never computes the
    full grid). The L2C companion carries no absolute azimuths, so only
    sza/vza/raa are checked here.
    """
    sub = {str(a): ds[str(a)].isel({str(names.rows): slice(None, None, 50),
                                    str(names.columns): slice(None, None, 50)})
           for a in (names.sza, names.vza, names.raa)}
    for a in (names.sza, names.vza, names.raa):
        lo, hi = float(a.attrs['minv']), float(a.attrs['maxv'])
        v = sub[str(a)].values
        assert v.min() >= lo - 1e-3 and v.max() <= hi + 1e-3, \
            f'{a} out of range [{lo}, {hi}], got [{v.min()}, {v.max()}]'


@pytest.fixture(autouse=True)
def _gc():
    # Dask graphs contain reference cycles; force a collection after every
    # test so reader datasets are released promptly.
    yield
    gc.collect()


def _write_l2c_companion(l1: Path) -> Path:
    """Stage the Level2C companion next to the real L1 granule.

    The archive does not hold one for this product; the reader only needs its
    three geometric grids, so constant values stand in (sza from the L1
    global sun angle). Idempotent: an existing file is left untouched, which
    also lets a real L2C dropped next to the granule take over.
    """
    l2c = l1.with_name(l1.name.replace('PRS_L1_STD_OFFL_', 'PRS_L2C_STD_', 1))
    if l2c.exists():
        return l2c
    with h5py.File(l1, 'r') as f:
        cube = f['HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube']
        ny, nx = cube.shape[0], cube.shape[2]
        raw_sza = f.attrs['Sun_zenith_angle'] if 'Sun_zenith_angle' in f.attrs else 30.0
        sza = float(np.asarray(raw_sza).item())
    with h5py.File(l2c, 'w') as f:
        geom = f.create_group('HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields')
        for name, value in [('Solar_Zenith_Angle', sza),
                            ('Observing_Angle', 15.0),
                            ('Rel_Azimuth_Angle', 45.0)]:
            geom.create_dataset(name, data=np.full((ny, nx), value, dtype='float32'))
    return l2c


@pytest.fixture(scope='session')
def prisma_l1():
    """Real L1 granule from the registry (clean skip when unavailable)."""
    return get_sample(Level1_PRISMA)


@pytest.fixture(scope='session')
def prisma_scene(prisma_l1, tmp_path_factory):
    """The real L1 granule staged next to its synthesized Level2C companion."""
    scene = tmp_path_factory.mktemp('prisma') / Path(prisma_l1).name
    scene.symlink_to(prisma_l1)
    _write_l2c_companion(scene)
    return scene


@pytest.fixture(scope='session')
def prisma(prisma_scene):
    return Level1_PRISMA(prisma_scene, chunks=500)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------

def test_l1_instantiation(prisma_scene, chunks):
    Level1_PRISMA(prisma_scene, chunks=chunks)


def test_l1_main(prisma):
    # PRISMA emits sza/vza/raa (no absolute azimuths in the L2C)
    generic.Test.main(prisma, angle_data=False)


def test_l1_conformance(prisma):
    generic.Test.v2_conformance(
        prisma, primary=str(names.ltoa),
        required_vars=(str(names.sza), str(names.vza), str(names.raa),
                       'PIXEL_SAT_ERR_MATRIX'))


def test_l1_latlon(prisma):
    # Orientation check only: the swath is a tilted parallelogram (latitude
    # varies along raster rows too), so the flat-projection invariance check
    # does not apply.
    generic.Test.latlon(prisma, flat_proj=False)


def test_l1_angles(prisma):
    _check_angles(prisma)


def test_l1_lazy_load(prisma):
    generic.Test.lazy_load(prisma)


def test_l1_flagreader(prisma):
    generic.Test.flagreader(prisma)
    # Computed check on a small subset (the full-grid isnull over the 234-band
    # cube is too heavy for the suite): flags must reproduce the per-mask
    # decodes bit-for-bit.
    sub = prisma.isel({str(names.rows): slice(0, 64), str(names.columns): slice(0, 64)})
    result = FlagsInit({GenericFlags.LAND: 1, GenericFlags.CLOUD: 2,
                        GenericFlags.L1_INVALID: 4}, 'uint8',
                       prisma.attrs['_flag_reader']).map_blocks(sub)
    flags = result['flags'].compute().values
    land = sub['LandCover_Mask'].compute().values > 0
    cloud = sub['Cloud_Mask'].compute().values == 1
    invalid = sub[str(names.ltoa)].isnull().any(dim=str(names.bands)).compute().values
    np.testing.assert_array_equal(flags & 1, land.view(np.uint8))
    np.testing.assert_array_equal(flags & 2, cloud.view(np.uint8))
    np.testing.assert_array_equal(flags & 4, invalid.view(np.uint8))


def test_l1_subset(prisma_scene):
    ds = Level1_PRISMA(prisma_scene, chunks=500)
    generic.Test.subset(ds)


# --------------------------------------------------------------------------
# Payload / bands
# --------------------------------------------------------------------------

def test_l1_bands(prisma, prisma_l1):
    # Band labels are the int-truncated central wavelengths (v1 convention),
    # ascending within each spectral group, invalid bands dropped
    with h5py.File(prisma_l1, 'r') as f:
        expected = {}
        for key in ('vnir', 'swir'):
            cw = np.asarray(f.attrs[f'List_Cw_{key.title()}'], dtype='float64')
            idx = np.where(cw > 0)[0][::-1]
            expected[key] = [str(int(c)) for c in cw[idx]]
    labels = list(prisma.bands.values)
    assert labels == expected['vnir'] + expected['swir']
    assert len(labels) == len(set(labels)), 'band labels must be unique'

    bgroups = prisma[str(names.bgroup)].values
    # Groups are contiguous blocks: all VNIR before all SWIR
    assert list(bgroups) == ['bands_vnir'] * len(expected['vnir']) + \
        ['bands_swir'] * len(expected['swir'])

    cwav = prisma['cwav'].values
    assert cwav.dtype == np.dtype('float32')
    for group in ('bands_vnir', 'bands_swir'):
        sel = cwav[bgroups == group]
        assert (np.diff(sel) > 0).all(), f'cwav not ascending within {group}'


def test_l1_units(prisma):
    assert prisma[str(names.ltoa)].attrs['units'] == 'W/m^2/um/sr'


def test_l1_raw_parity(prisma, prisma_l1):
    """Ltoa reproduces the raw uint16 decode at the mirrored columns.

    The sample is stored column 0 = eastern edge, so the per-file flip makes
    v2 column j carry raw column nx-1-j — pinned here against the undecoded
    file values, for one band of each spectral group.
    """
    nx = prisma.sizes[str(names.columns)]
    y0, y1 = 100, 130          # rows (not flipped in this scene)
    r0, r1 = 300, 340          # raw column window in the native layout
    with h5py.File(prisma_l1, 'r') as f:
        for key in ('vnir', 'swir'):
            Key = key.upper()
            cw = np.asarray(f.attrs[f'List_Cw_{key.title()}'], dtype='float64')
            scale = float(np.asarray(f.attrs[f'ScaleFactor_{key.title()}']).item())
            offset = float(np.asarray(f.attrs[f'Offset_{key.title()}']).item())
            cube = f[f'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/{Key}_Cube']

            idx = np.where(cw > 0)[0][::-1]
            file_idx = int(idx[len(idx) // 2])   # middle output band of the group
            label = str(int(cw[file_idx]))

            # v2 column j carries raw column nx-1-j, so the raw window
            # [r0, r1) maps to v2 x in [nx-r1, nx-r0) with reversed order.
            raw = cube[y0:y1, file_idx, r0:r1].astype('float64')
            want = np.where(raw < 65535, (raw - offset) / scale, np.nan)
            want = want.astype('float32')[:, ::-1]

            got = prisma[str(names.ltoa)].sel(bands=label).isel(
                y=slice(y0, y1), x=slice(nx - r1, nx - r0)
            ).compute().values
            assert got.dtype == np.dtype('float32')
            np.testing.assert_array_equal(got, want)


# --------------------------------------------------------------------------
# Attributes
# --------------------------------------------------------------------------

def test_l1_attributes(prisma, prisma_scene):
    a = prisma.attrs
    assert a[str(names.sensor)] == 'PRISMA'
    assert a[str(names.platform)] == 'PRISMA'
    # Nominal 30 m pixel size (ATBD); v1 hardcoded the same value
    assert a[str(names.resolution)] == 30
    assert a[str(names.product_name)] == Path(prisma_scene).name
    assert a[str(names.input_directory)] == str(Path(prisma_scene).parent)
    assert a[str(names.datetime)] == '2021-07-21T10:27:00.923000'
    assert a[str(names.shortname)] == 'PRS_L1_STD'
    assert a[str(names.crs)] == 'EPSG:4326'
    assert a['_flag_reader'] == 'eoread.readers.prisma.Level1_PRISMA'
    # Full product metadata carried as an attribute (bytes decoded to str)
    md = a['metadata']
    assert isinstance(md, dict) and md['Product_ID'] == 'PRS_L1_STD'


def test_l1_v1_compat(prisma):
    old = v1(prisma)
    # identity + schema_version strip (no PRISMA renames or clean breaks)
    assert str(names.ltoa) in old
    assert 'schema_version' not in old.attrs


# --------------------------------------------------------------------------
# Companion / variant handling
# --------------------------------------------------------------------------

def test_l1_l2c_required(tmp_path):
    l1 = tmp_path / 'PRS_L1_STD_OFFL_20220423112851_20220423112856_0001.he5'
    l1.touch()
    with pytest.raises(ProductError, match='L2C companion'):
        Level1_PRISMA(l1)


def _write_fake_product(directory, product_id=b'PRS_L1_STD'):
    """Write a minimal PRISMA L1+L2C product pair for unit testing."""
    ny, nx = 8, 8

    # VNIR: 3 file bands (descending cw), last one invalid (cw == 0)
    vnir = np.full((ny, 3, nx), 100, dtype='uint16')
    vnir[:, 0, :] = 12345
    vnir[3, 1, 4] = 65535  # invalid pixel in file band 1 (cw 400)
    swir = np.full((ny, 2, nx), 200, dtype='uint16')

    l1 = directory / 'PRS_L1_STD_OFFL_20220101000000_20220101000005_0001.he5'
    with h5py.File(l1, 'w') as f:
        f.attrs['Product_ID'] = product_id
        f.attrs['Product_StartTime'] = b'2022-01-01T00:00:00.000000'
        f.attrs['List_Cw_Vnir'] = np.array([500.0, 400.0, 0.0])
        f.attrs['List_Fwhm_Vnir'] = np.array([10.0, 10.0, 0.0])
        f.attrs['ScaleFactor_Vnir'] = 100.0
        f.attrs['Offset_Vnir'] = 0.0
        f.attrs['List_Cw_Swir'] = np.array([1500.0, 1400.0])
        f.attrs['List_Fwhm_Swir'] = np.array([10.0, 10.0])
        f.attrs['ScaleFactor_Swir'] = 100.0
        f.attrs['Offset_Swir'] = 0.0
        data = f.create_group('HDFEOS/SWATHS/PRS_L1_HCO/Data Fields')
        data.create_dataset('VNIR_Cube', data=vnir)
        data.create_dataset('SWIR_Cube', data=swir)
        data.create_dataset('VNIR_PIXEL_SAT_ERR_MATRIX', data=np.zeros((ny, 3, nx), dtype='uint8'))
        data.create_dataset('SWIR_PIXEL_SAT_ERR_MATRIX', data=np.zeros((ny, 2, nx), dtype='uint8'))
        data.create_dataset('LandCover_Mask', data=np.ones((ny, nx), dtype='uint8'))
        data.create_dataset('Cloud_Mask', data=np.zeros((ny, nx), dtype='uint8'))
        geo = f.create_group('HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields')
        geo.create_dataset('Latitude_VNIR', data=np.full((ny, nx), 40.0, dtype='float32'))
        geo.create_dataset('Longitude_VNIR', data=np.full((ny, nx), -8.0, dtype='float32'))

    l2 = directory / 'PRS_L2C_STD_20220101000000_20220101000005_0001.he5'
    with h5py.File(l2, 'w') as f:
        geom = f.create_group('HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields')
        for name in ['Solar_Zenith_Angle', 'Observing_Angle', 'Rel_Azimuth_Angle']:
            geom.create_dataset(name, data=np.full((ny, nx), 30.0, dtype='float32'))

    return l1


def test_raw_invalid_masking(tmp_path):
    """Raw fill value (65535) must be masked to NaN in the raw domain."""
    ds = Level1_PRISMA(_write_fake_product(tmp_path), chunks=4)

    # Bands re-ordered ascending, invalid band (cw == 0) dropped
    assert list(ds['bands'].values) == ['400', '500', '1400', '1500']
    assert list(ds['cwav'].values) == [400.0, 500.0, 1400.0, 1500.0]

    ltoa = ds['Ltoa'].compute()
    # Scaling applied (raw / ScaleFactor)
    assert np.allclose(ltoa.sel(bands='500').values, 123.45)
    assert np.allclose(ltoa.sel(bands='1400').values, 2.0)
    # Invalid raw pixel masked to NaN (and not scaled to 655.35)
    band = ltoa.sel(bands='400').values
    assert np.isnan(band[3, 4])
    assert np.allclose(band[~np.isnan(band)], 1.0)

    # L1_INVALID flag set on the masked pixel only
    flags = Level1_PRISMA.getflag(ds, GenericFlags.L1_INVALID).compute()
    assert int(flags.sum()) == 1
    assert bool(flags.isel(y=3, x=4))


def test_fwhm(tmp_path):
    """The band widths are provided, with the central wavelengths"""
    ds = Level1_PRISMA(_write_fake_product(tmp_path), verbose=False)
    assert list(ds.fwhm.values) == [10., 10., 10., 10.]
    assert ds.fwhm.dims == ds.cwav.dims == ('bands',)


def test_l1_rejects_non_std(tmp_path):
    # A root-level Product_ID other than PRS_L1_STD must be rejected loudly.
    bad = _write_fake_product(tmp_path, product_id=b'PRS_L2C_STD')
    with pytest.raises(UnsupportedProductType):
        Level1_PRISMA(bad)
