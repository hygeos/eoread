from functools import wraps
import json
from pathlib import Path
import pickle
import xarray as xr
from eoread import eo


def cachefunc(cache_file, method='json'):
    """
    A decorator that caches the return of a function in a file
    
    method: 'json' or 'pickle'
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if Path(cache_file).exists():
                if method == 'json':
                    with open(cache_file) as fp:
                        return json.load(fp)
                elif method == 'pickle':
                    with open(cache_file, 'rb') as fp:
                        return pickle.load(fp)
                else:
                    raise ValueError
            else:
                result = f(*args, **kwargs)
                Path(cache_file).parent.mkdir(exist_ok=True, parents=True)
                if method == 'json':
                    with open(cache_file, 'w') as fp:
                        json.dump(result, fp, indent=4)
                elif method == 'pickle':
                    with open(cache_file, 'wb') as fp:
                        pickle.dump(result, fp)
                else:
                    raise ValueError
                return result
        return wrapper
    return decorator
        

def cache_dataset(cache_file, attrs=None):
    """
    A decorator that caches the dataset returned by a function in a netcdf file
    
    The attribute dictionary `attrs` is stored in the file, and verified upon reading
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if Path(cache_file).exists():
                print(f'Using cache file {cache_file}')
                ds = xr.open_dataset(cache_file)

                # check attributes in loaded file
                if attrs is not None:
                    for k, v in attrs.items():
                        assert ds.attrs[k] == v, \
                            f'Error when checking attribute {k}: {ds.attrs[k]} != {v}'

            else:
                ds = f(*args, **kwargs)
                Path(cache_file).parent.mkdir(exist_ok=True, parents=True)

                # store provided attributes in cache file
                if attrs is not None:
                    ds.attrs.update(attrs)

                # Write `ds` to `cache_file`
                eo.to_netcdf(ds, filename=cache_file)
                print(f'Wrote cache file {cache_file}')
                
            return ds
        return wrapper
    return decorator
    