import json
import pickle
import pandas as pd
import xarray as xr

from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal, Optional
from pandas.testing import assert_frame_equal

from .save import to_netcdf
from .fileutils import filegen, safe_move


def cachefunc(cache_file: Path|str,
              reader: Callable,
              writer: Callable,
              check_in: Optional[Callable] = None,
              check_out: Optional[Callable] = None,
              fg_kwargs=None):
    """
    A decorator that caches the return of a function in a file, with
    customizable format

    reader: a function that reads inputs/output from the cache file
        reader(filename) -> {'output': ..., 'input': ...}

    writer: a function that writes the inputs/output to the cache file
        writer(filename, output, input_args, input_kwargs)

    check_in: a custom function to test the equality of the inputs
        checker(obj1, obj2) -> bool
        (defaults to None -> no checking)
    
    check_out: a custom function to test the equality of the outputs
        checker(obj1, obj2) -> bool
        (defaults to ==)

    fg_kwargs: kwargs passed to filegen (ex: lock_timeout=-1)
    """
    # default input/output checkers
    check_out = check_out or (lambda x, y: x == y)

    cache_file = Path(cache_file)
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not cache_file.exists():

                result = f(*args, **kwargs)

                with TemporaryDirectory(dir=cache_file.parent,
                                        prefix='cachefunc_') as tmpdir:
                    cache_file_tmp = Path(tmpdir)/cache_file.name

                    # write the temporary cache file
                    filegen(**(fg_kwargs or {}))(writer)(
                        cache_file_tmp, result, args, kwargs
                    )

                    # check that the output read back is identical
                    # to the original output (defaults to ==)
                    chk = reader(cache_file_tmp)
                    assert 'output' in chk

                    assert check_out(result, chk['output'])

                    # check successful: move the file
                    safe_move(cache_file_tmp, cache_file)
                    assert cache_file.exists()

                    return chk['output']
            else:
                content = reader(cache_file)

                # check function inputs
                if check_in:
                    with TemporaryDirectory() as tmpdir:
                        # write and read back the inputs for checking
                        tmpchk = Path(tmpdir)/'tmp_check'
                        writer(tmpchk, None, args, kwargs)
                        chk = reader(tmpchk)
                        if not check_in(content['input'], chk['input']):
                            raise ValueError(
                                f"Input parameters do not match stored inputs.\n"
                                f"  Input params:  {content['input']}\n"
                                f"  Stored inputs: {chk['input']}"
                            )

                return content['output']

        return wrapper
    return decorator


def cache_dataframe(cache_file: Path | str):
    return cachefunc(
        cache_file,
        writer=lambda filename, df, args, kwargs: df.to_csv(filename, index=False),
        reader=lambda filename: {"output": pd.read_csv(filename, parse_dates=["time"])},
        check_out=lambda x, y: assert_frame_equal,
    )


def cache_json(
    cache_file: Path | str,
    inputs: Literal["check", "store", "ignore"] = "check",
):
    """
    A decorator that caches the result of a function to a json file.

    inputs:
        "check" [default]: store and check the function inputs
        "store": store but don't check the function inputs
        "ignore": ignore the function inputs
    """

    def reader(filename):
        with open(filename) as fp:
            return json.load(fp)

    def writer(filename, output, input_args, input_kwargs):
        with open(filename, 'w') as fp:
            content = {}
            if inputs in ['store', 'check']:
                content['input'] = {'args': input_args, 'kwargs': input_kwargs}
            content['output'] = output
            json.dump(content, fp, indent=4, default=str)

    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
        check_in=(lambda x, y : x == y) if (inputs == 'check') else None,
    )


def cache_pickle(
    cache_file: Path | str,
    inputs: Literal["check", "store", "ignore"] = "check",
):

    def reader(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    def writer(filename, output, input_args, input_kwargs):
        with open(filename, 'wb') as fp:
            content = {}
            content['input'] = {'args': input_args, 'kwargs': input_kwargs}
            content['output'] = output
            pickle.dump(content, fp)
    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
        check_in=(lambda x, y : x == y) if (inputs == 'check') else None,
    )


def cache_dataset(cache_file: Path|str,
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
        return {'output': ds}

    def writer(filename, ds_out, input_args, input_kwargs):
        if attrs is not None:
            ds_out.attrs.update(attrs)
        to_netcdf(ds_out, filename=filename)

    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
    )
