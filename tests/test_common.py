#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import pytest
import xarray as xr
import numpy as np
import dask.array as da
from eoread.common import AtIndex, Repeat
from eoread.common import Interpolator, ceil_dt, floor_dt
from eoread.common import DataArray_from_array, timeit
from eoread.reader import msi
from eoread.reader.gsw import GSW
from eoread.utils.fileutils import PersistentList
from eoread.utils.interpolate import selinterp
from eoread.utils.tools import split, merge, wrap, raiseflag, convert, locate, xrcrop
from time import sleep
from pathlib import Path
from tempfile import TemporaryDirectory
import dask
from . import conftest

dask.config.set(scheduler='single-threaded')

class VerboseArray:
    def __init__(self, A, name=None, verbose=False):
        self._A = A
        self.shape = A.shape
        self.dtype = A.dtype
        self.ndim = A.ndim
        self._name = name
        self._verbose = verbose

    def __getitem__(self, keys):
        if self._verbose:
            print('Reading', self._name, keys)
        return self._A[keys]


def make_dataset(shp=(10, 12), chunks=6):
    l1 = xr.Dataset()
    bands = [412, 443, 490, 510, 560, 620, 865]
    shp2 = shp
    shp3 = (len(bands),) + shp
    dims2 = ('x', 'y')
    dims3 = ('bands', 'x', 'y')

    for (name, s, dims, chk) in [
            ('rho_toa', shp3, dims3, (-1, chunks, chunks)),
            ('rho_w', shp3, dims3, (-1, chunks, chunks)),
            ('lat', shp2, dims2, (chunks, chunks)),
            ('lon', shp2, dims2, (chunks, chunks)),
        ]:
        l1[name] = DataArray_from_array(
            VerboseArray(
                np.random.random(s),
                name=name,
                verbose=True,
            ),
            dims=dims,
            chunks=chk,
            )
    l1 = l1.assign_coords(bands=bands)

    # set some attributes
    l1.attrs['sensor'] = 'OLCI'
    l1.rho_w.attrs['unit'] = 'dimensionless'

    return l1


def test_split():
    l1 = make_dataset()
    print(l1)
    l1s = split(l1, 'bands')
    assert 'rho_toa_412' in l1s
    assert 'rho_w_412' in l1s
    assert 'rho_toa' not in l1s
    assert 'sensor' in l1s.attrs
    assert 'unit' in l1s.rho_w_412.attrs
    assert 'bands' in l1s.coords   # split should preserve coords
    print(l1s)


def test_split_dataarray():
    l1 = make_dataset()
    l1s = split(l1.rho_w, 'bands')
    print(l1s)


def test_split_without_coords():
    l1 = make_dataset()

    with pytest.raises(AssertionError):
        split(l1.rho_w, 'unknown_dim')

    l1 = l1.drop('bands')
    with pytest.raises(AssertionError):
        split(l1.rho_w, 'bands')


def test_merge():
    """
    Merge many variables
    """
    # create a dataset
    l1 = make_dataset()
    l1s = split(l1, 'bands')
    l1m = merge(l1s)
    print('Original:', l1)
    print('Merged:', l1m)
    assert l1m.equals(l1)

def test_merge2():
    """
    Merge a single variable
    """
    l1 = make_dataset()
    l1s = split(l1, 'bands')
    l1m = merge(l1s,
        dim='bands',
        varname='rho_w',
        pattern=r'rho_w_(\d+)')
    print('Original:', l1)
    print('Merged:', l1m)


def test_merge_inconsistent_dimension():
    l1 = make_dataset()
    l1s = split(l1, 'bands')
    merge(l1s, 'bands')
    l1s = l1s.rename({'rho_w_412': 'rho_w_413'})
    with pytest.raises(AssertionError):
        merge(l1s, 'bands')


@pytest.mark.parametrize('A,rep', [
    (np.random.rand(5), (3,)),
    (np.random.rand(10, 15), (3, 4)),
    (da.eye(5, chunks=2), (3, 3)),
], ids=[
    'rand.1D',
    'rand.2D',
    'dask.eye.2D',
    ])
@pytest.mark.parametrize('idx1', [
    slice(None),
    slice(None, None, 2),
    slice(None, None, 3),
    slice(None, None, 5),
    slice(4, 11, 3),
    slice(4, 11),
    slice(4, 11, 2),
    slice(None, None, 6),
    slice(None, 2),
    slice(1, 2),
    slice(2, -2),
    slice(2, None),
    slice(2, None, 3),
    slice(2, -2, 3),
    slice(4, -2, 3),
    slice(2, -2, 6),
    slice(4, -2, 6),
    0,
    1,
    -1,
    -2,
    -3,
])
@pytest.mark.parametrize('idx2', [0, slice(None)])
def test_repeat(A, rep, idx1, idx2):
    B = Repeat(A, rep)
    BB = A.copy()
    for i, r in enumerate(rep):
        BB = BB.repeat(r, axis=i)

    if A.ndim == 1:
        idx = idx1
    else:
        idx = (idx1, idx2)

    np.testing.assert_allclose(B[idx], BB[idx])


def test_interpolator():
    A = xr.DataArray(
        np.eye(5),
        dims=('x', 'y'),
        coords={
            'x': np.arange(5)*2,
            'y': np.arange(5)*2,
        })
    I = Interpolator((10, 10), A)
    assert I[0, 0] == 1.
    assert I[1, 0] == 0.5


def test_da_from_array_meta():
    """
    Check that da.from_array has an argument `meta` (use a recent version of dask)

    Note: this argument avoids calling A.__getitem__ to retrieve its dtype
    """
    A = np.eye(10)
    da.from_array(
        A,
        chunks=(2, 2),
        meta=np.array([], A.dtype),
    )


def test_AtIndex_1():
    shp = (25, 25)
    A = xr.DataArray(
        np.random.random(100),
        dims=('index'),
    )
    idx = xr.DataArray(
        np.random.randint(0, 100, shp),
        dims=('x', 'y'),
    )

    AA = AtIndex(
        A,
        idx,
        'index'
    )

    assert AA.dims == ('x', 'y')
    assert AA.shape == shp
    np.testing.assert_allclose(AA[:, :], A[idx])


def test_AtIndex_2():
    shp = (25, 25)
    A = xr.DataArray(
        np.random.random((10, 100)),
        dims=('bands', 'index'),
    )
    idx = xr.DataArray(
        np.random.randint(0, 100, shp),
        dims=('x', 'y'),
    )

    AA = AtIndex(
        A,
        idx,
        'index'
    )

    assert AA.dims == ('bands', 'x', 'y')
    assert AA.shape == (10,) + shp
    np.testing.assert_allclose(AA[:, :, :], A[:, idx])


@pytest.mark.parametrize('use_dask', [True, False])
def test_raiseflag(use_dask):
    shp = (13, 7)
    flags = xr.DataArray(
        da.zeros(shp,
                 dtype='uint16',
                 chunks=5)
    )

    # raise a flag using binary
    A = xr.DataArray(
        da.random.random_integers(0, 10, size=shp, chunks=5)
    )
    if not use_dask:
        A = A.compute()
        flags = flags.compute()

    assert (A & 3).any()
    assert not flags.any()
    raiseflag(flags, 'FLAG_1', 1, A & 3)
    assert flags.any()

    # raising a second flag with same value raises an error
    with pytest.raises(AssertionError):
        raiseflag(flags, 'FLAG_2', 1, A > 0)

    # raising the same flag with a different value raises an error
    with pytest.raises(AssertionError):
        raiseflag(flags, 'FLAG_1', 2, A > 0)

    # raise a second flag
    raiseflag(flags, 'FLAG_2', 4, A > 1)

    assert (flags > 0).any()


def test_floor_ceil_dt():
    dt = datetime(2020, 3, 3, 14, 26, 48)
    delta = timedelta(minutes=15)
    assert floor_dt(dt, delta) == datetime(2020, 3, 3, 14, 15, 00)
    assert ceil_dt(dt, delta) == datetime(2020, 3, 3, 14, 30, 00)


@pytest.mark.parametrize('vmin2,vmax2', [
    (-180, 180),
    (0, 360),
])
@pytest.mark.parametrize('vmin1,vmax1', [
    (0, 360),
    (-180, 180),
])
def test_wrap(vmin1,vmax1,vmin2,vmax2):
    print(f'[{vmin1},{vmax1-1}] -> [{vmin2},{vmax2}]')
    ds = xr.Dataset()
    ds['A'] = xr.DataArray(np.arange(vmin1, vmax1),
                           dims=['lon'],
                           coords=[np.arange(vmin1, vmax1)])
    print(ds.lon)
    res = wrap(ds, 'lon', vmin2, vmax2)
    print(res.lon)

    assert (np.diff(res.lon) > 0).all()
    assert (res.lon >= vmin2).all()
    assert (res.lon <= vmax2).all()


def test_convert():
    A = xr.DataArray(300., attrs={'units': 'DU'})
    convert(A, unit_to='kg/m2')

    A = xr.DataArray(1., attrs={'units': 'kg'})
    assert convert(A, unit_to='g', converter={'g': 1000, 'kg': 1}) == 1000.

    with pytest.raises(ValueError):
        convert(A, unit_to='g', converter={})

def test_locate():
    lat_min = 0
    lat_max = 10
    lon_min = 0
    lon_max = 10
    lat, lon = xr.broadcast(
        xr.DataArray(np.linspace(lat_min, lat_max, 100), dims=['lat']),
        xr.DataArray(np.linspace(lon_min, lon_max, 100), dims=['lon']),
    )
    locate(lat, lon, 5., 5.)
    locate(lat, lon, 15., 15.)

    locate(lat, lon, 5., 5., dist_min_km=10)
    with pytest.raises(ValueError):
        locate(lat, lon, 15., 15., dist_min_km=10)


@pytest.mark.parametrize('concurrent', [True, False])
def test_persistent_list(concurrent):
    with TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir)/'list.json'
        a = [1, 2, "three"]

        A = PersistentList(filename, concurrent=concurrent)
        A.extend(a)
        assert 1 in A
        assert len(A) == 3

        B = PersistentList(filename, concurrent=concurrent)
        assert A == B
        B.append('4')

        C = PersistentList(filename, concurrent=concurrent)
        assert len(C) == 4
        C.clear()
        assert len(C) == 0

def test_persistent_list_concurrent():
    with TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir)/'list.json'
        A = PersistentList(filename)
        B = PersistentList(filename)
        A.append(1)
        assert len(B) == 1
        assert 1 in B
        B.append(2)
        assert len(A) == 2
        assert len(B) == 2


def test_timeit_decorator():
    @timeit(desc='f')
    def f():
        sleep(1)
    f()


def test_timeit_context():
    with timeit() as ti:
        sleep(1)
    assert ti() > 1


@pytest.mark.parametrize('idx', [
    np.array([10, 11, 12, 13]),
    np.array([13, 12, 11, 10]),
])
@pytest.mark.parametrize('min_max,expected_len', [
    ((11.5, 12.5), 3),
    ((11, 12), 2),
    ((9, 14), 4),
])
def test_xrcrop(idx, min_max, expected_len):
    """
    Test xrcrop on basic data
    """
    full = xr.DataArray(np.zeros(len(idx)),
                        dims=["x"],
                        coords={"x": idx})
    cropped = xrcrop(full, x=min_max)
    print('Initial index is  ', idx)
    print('Min max values are', min_max)
    print('Cropped values are', cropped.x.values)
    assert len(cropped.x) == expected_len


@pytest.mark.parametrize('method', ["nearest", "interp"])
def test_xrcrop_gsw(request, method):
    """
    Test the application of xrcrop on gsw.
    This allows .computing the result.
    Otherwise, the further .sel is very slow over large arrays.
    """
    ds = msi.Level1_MSI(msi.get_sample())

    plt.figure()
    ds.Rtoa.sel(bands=865).plot.imshow(origin='upper')
    conftest.savefig(request)

    gsw = GSW(agg=4)
    with timeit("Optimize"):
        gsw = xrcrop(
            gsw,
            latitude=ds.latitude,
            longitude=ds.longitude,
        )

    with timeit("Compute gsw"):
        gsw = gsw.compute()

    with timeit("sel"):
        data = selinterp(
            gsw,
            latitude=ds.latitude,
            longitude=ds.longitude,
            method=method,
        )
    with timeit("compute"):
        print(data.dtype)
        data = data.compute()
        print(data.dtype)

    plt.figure()
    plt.imshow(data, origin="upper")
    conftest.savefig(request)
