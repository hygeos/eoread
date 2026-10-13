from core.geo.naming import names
from numpy.random import randint
from eoread.tools import *
import xarray as xr
import numpy as np
import pytest

@pytest.fixture
def metadata(): return {'a':0, 'b': {'c':1, 'd':2}}

@pytest.fixture(params=[(6,6),(20,30),(500,500)])
def shape(request): return request.param

@pytest.fixture
def array(shape): return xr.DataArray(randint(0,10,shape))


def test_filter_metadata(metadata):
    dico = filter_metadata(metadata, ['a',['b','d']])
    assert dico['b'].get('c') is None

def test_filter_metadata_basic(metadata):
    template = [['a'], ['b','c']]
    result = filter_metadata(metadata, template)
    assert result == {'a': 0, 'b': {'c': 1}}

def test_filter_metadata_missing_key(metadata):
    template = [['a'], ['b','x']]
    with pytest.raises(ValueError):
        filter_metadata(metadata, template)

@pytest.mark.parametrize('method', ['linear','repeat'])
def test_spatial_resample(array, method):
    dims = dict(dim_0=50, dim_1=50)
    chunks = dict(dim_0=10, dim_1=10)
    arr = spatial_resample(array, dims, chunks, method)
    assert arr.shape == (50,50)

def test_spatial_resample_identity(array):
    dims = {d: array.shape[i] for i, d in enumerate(array.dims)}
    chunks = {d: array.shape[i] for i, d in enumerate(array.dims)}
    arr = spatial_resample(array, dims, chunks)
    assert arr.shape == array.shape

def test_spatial_resample_downsample(array):
    dims = {d: 2 for d in array.dims}
    chunks = {d: 2 for d in array.dims}
    arr = spatial_resample(array, dims, chunks)
    assert arr.shape == (2, 2)

def test_spatial_resample_oversample(array):
    dims = {d: 10 for d in array.dims}
    chunks = {d: 10 for d in array.dims}
    arr = spatial_resample(array, dims, chunks)
    assert arr.shape == (10, 10)

def test_format_chunks():
    # int input
    assert format_chunks(5) == {str(names.rows): 5, str(names.columns): 5}
    # list input
    assert format_chunks([2, 3]) == {str(names.rows): 2, str(names.columns): 3}
    # tuple input
    assert format_chunks((4, 6)) == {str(names.rows): 4, str(names.columns): 6}
    # dict input
    d = {str(names.rows): 7, str(names.columns): 8}
    assert format_chunks(d) == d
    # error: wrong length
    with pytest.raises(AssertionError):
        format_chunks([1])
    # error: missing keys
    with pytest.raises(AssertionError):
        format_chunks({'rows': 1, 'foo': 2})

def test_open_raster(tmp_path):
    import xarray as xr
    arr = xr.DataArray([[1,2],[3,4]], dims=("rows","columns"))
    f = tmp_path / "test.nc"
    arr.to_netcdf(f, engine='h5netcdf')
    # Should find and open the raster
    out = open_raster(tmp_path, "*.nc")
    assert out.shape == (2,2)
    assert np.allclose(out.values, arr.values)
    # Should fail if multiple files
    (tmp_path / "test2.nc").write_bytes(f.read_bytes())
    with pytest.raises(ValueError):
        open_raster(tmp_path, "*.nc")
    # Should fail if no files
    with pytest.raises(ValueError):
        open_raster(tmp_path, "*.h5")