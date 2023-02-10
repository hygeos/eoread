from pathlib import Path
import pytest
from eoread.cache import cachefunc, cache_dataset
from xarray.tutorial import open_dataset
from tempfile import TemporaryDirectory

@pytest.mark.parametrize('var', [
    1,
    'a',
    [1, 'b', [1, 2]],
])
def test_cachefunc(var):
    def my_function():
        return var
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/'cache.json'
        cachefunc(cache_file)(my_function)()
        a = cachefunc(cache_file)(my_function)()
        assert a == my_function()

def test_cache_dataset():
    def my_function():
        return open_dataset('air_temperature')
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/'cache.nc'
        cache_dataset(cache_file)(my_function)()
        a = cache_dataset(
            cache_file,
            chunks={'lat': 10, 'lon': 10},
        )(my_function)()
        assert a == my_function()
    