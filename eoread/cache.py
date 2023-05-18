from functools import wraps
import json
from pathlib import Path
import pickle
from typing import Callable, Optional
import xarray as xr
from eoread import eo
from eoread.fileutils import filegen


def cachefunc(cache_file: Path,
              reader: Callable,
              writer: Callable,
              checker: Optional[Callable] = None,
              fg_kwargs=None):
    """
    A decorator that caches the return of a function in a file, with
    customizable format

    writer: a function
        obj = reader(filename)
    reader: a function
        writer(filename, obj)
    checker: a custom function to test the equality of the two objects
        checker(obj1, obj2)
        (defaults to ==)
    fg_kwargs: kwargs passed to filegen (ex: lock_timeout=-1)
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not cache_file.exists():
                result = f(*args, **kwargs)

                filegen(**(fg_kwargs or {}))(writer)(cache_file, result)
            else:
                result = None

            obj = reader(cache_file)

            if result is not None:
                # check that the object read back is identical
                # to the original result (defaults to ==)
                if checker is None:
                    assert result == obj
                else:
                    assert checker(result, obj)

            return obj

        return wrapper
    return decorator


def cache_json(cache_file: Path):

    def reader(filename):
        with open(filename) as fp:
            return json.load(fp)

    def writer(filename, obj):
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)

    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
    )


def cache_pickle(cache_file: Path):

    def reader(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    def writer(filename, obj):
        with open(filename, 'wb') as fp:
            pickle.dump(obj, fp)
    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
    )


def cache_dataset(cache_file,
                  attrs=None,
                  **kwargs):
    """
    A decorator that caches the dataset returned by a function in a netcdf file

    The attribute dictionary `attrs` is stored in the file, and verified upon
    reading.

    Other kwargs (ex: chunks) are passed to xr.open_dataset
    """
    def reader(filename):
        ds = xr.open_dataset(filename, **kwargs)

        # check attributes in loaded file
        if attrs is not None:
            for k, v in attrs.items():
                assert ds.attrs[k] == v, \
                    f'Error when checking attribute {k}: {ds.attrs[k]} != {v}'
        return ds

    def writer(filename, ds):
        if attrs is not None:
            ds.attrs.update(attrs)
        eo.to_netcdf(ds, filename=filename)

    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
    )
