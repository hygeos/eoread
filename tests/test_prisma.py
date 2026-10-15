#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import pytest

from eoread.flags import GenericFlags
from eoread.prisma import FlagsReader_PRISMA, Level1_PRISMA, get_sample
from . import generic


@pytest.fixture(scope='session')
def prisma_level1():
    return get_sample(1)


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param


@pytest.fixture
def PRISMA_product(chunks, prisma_level1):
    return Level1_PRISMA(prisma_level1, chunks=chunks)


def test_instantiation(chunks, prisma_level1):
    Level1_PRISMA(prisma_level1, chunks=chunks)


def test_main(PRISMA_product):
    generic.Test.main(PRISMA_product, angle_data=False)


def test_time(chunks, prisma_level1):
    params = {'filename': prisma_level1, 'chunks': chunks}
    generic.Test.execution_time(Level1_PRISMA, params)


def test_lazy_load(PRISMA_product):
    generic.Test.lazy_load(PRISMA_product)


def test_subset(PRISMA_product):
    generic.Test.subset(PRISMA_product)


def test_plot(request, prisma_level1):
    l1 = Level1_PRISMA(prisma_level1)
    band = min(l1.bands.values, key=lambda b: abs(int(b) - 865))
    generic.plot(request, l1, band, poi={'x': 500, 'y': 500})


def test_flag_reader(PRISMA_product):
    generic.Test.flagreader(PRISMA_product)


def test_bands_groups(PRISMA_product):
    groups = set(PRISMA_product.bands_group.values)
    assert groups == {'bands_vnir', 'bands_swir'}


def test_l2c_required(tmp_path):
    l1 = tmp_path / 'PRS_L1_STD_OFFL_20220423112851_20220423112856_0001.he5'
    l1.touch()
    with pytest.raises(AssertionError, match='L2C companion'):
        Level1_PRISMA(l1)


def _write_fake_product(directory):
    """Write a minimal PRISMA L1+L2C product pair for unit testing."""
    ny, nx = 8, 8

    # VNIR: 3 file bands (descending cw), last one invalid (cw == 0)
    vnir = np.full((ny, 3, nx), 100, dtype='uint16')
    vnir[:, 0, :] = 12345
    vnir[3, 1, 4] = 65535  # invalid pixel in file band 1 (cw 400)
    swir = np.full((ny, 2, nx), 200, dtype='uint16')

    l1 = directory / 'PRS_L1_STD_OFFL_20220101000000_20220101000005_0001.he5'
    with h5py.File(l1, 'w') as f:
        f.attrs['Product_StartTime'] = b'2022-01-01T00:00:00.000000'
        f.attrs['List_Cw_Vnir'] = np.array([500.0, 400.0, 0.0])
        f.attrs['ScaleFactor_Vnir'] = 100.0
        f.attrs['Offset_Vnir'] = 0.0
        f.attrs['List_Cw_Swir'] = np.array([1500.0, 1400.0])
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
    flags = FlagsReader_PRISMA().getflag(ds, GenericFlags.L1_INVALID).compute()
    assert int(flags.sum()) == 1
    assert bool(flags.isel(y=3, x=4))
