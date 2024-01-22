#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for saving results
'''

import shutil
import tempfile
import xarray as xr
import numpy as np

from pathlib import Path
from contextlib import contextmanager
from dask.diagnostics import ProgressBar

from .naming import naming


def to_netcdf(ds: xr.Dataset, *,
              filename: str = None,
              dirname: str = None,
              product_name: str = None,
              ext: str = '.nc',
              product_name_attr: str = 'product_name',
              if_exists: str = 'error',
              tmpdir: str = None,
              create_out_dir: str = True,
              engine: str = 'h5netcdf',
              zlib: bool = True,
              complevel: int = 5,
              verbose: bool = True,
              **kwargs):
    '''
    Write an xarray Dataset `ds` using `.to_netcdf` with several additional features:
    - construct file name using  `dirname`, `product_name` and `ext`
      (optionnally - otherwise with `filename`)
    - check that the output file does not exist already
    - Use file compression
    - Use temporary file
    - Create output directory if it does not exist

    Arguments:
    ----------

    filename: str or None
        Output file path.
        if None, build filename from `dirname`, `product_name` and `ext`.
    dirname: str or None
        directory for output file (default None: uses the attribute input_directory of ds)
    product_name: str
        base name for the output file. If None (default), use the attribute named attr_name
    ext: str
        extension (default: '.nc')
    product_name_attr: str
        name of the attribute to use for product_name in `ds`
    if_exists: 'error', 'skip' or 'overwrite'
        what to do if output file exists
    tmpdir: str ; default None = system directory
        use a given temporary directory
    create_out_dir: str
        create output directory if it does not exist

    Other kwargs are passed to `to_netcdf`

    About engine and compression:
        - Use default engine='h5netcdf' (much faster than 'netcdf4' when activating compression)
        - Use compression by default: encoding={'zlib':True, 'complevel':9}.
          Compression can be disactivated by passing encoding={}


    Returns:
    -------

    output file name (str)
    '''
    assert isinstance(ds, xr.Dataset), 'to_netcdf expects an xarray Dataset'
    if filename is None:
        # construct filename from dirname, product_name and ext
        if product_name is None:
            product_name = ds.attrs[product_name_attr]
        assert product_name, 'Empty product name'
        if dirname is None:
            dirname = ds.attrs[naming.input_directory]
        fname = Path(dirname).resolve()/(product_name+ext)

    else:
        fname = Path(filename).resolve()

    if fname.exists():
        if if_exists == 'skip':
            print(f'File {fname} exists, skipping...')
            return fname
        elif if_exists == 'overwrite':
            fname.unlink()
        elif if_exists == 'error':
            raise IOError(f'Output file "{fname}" exists.')
        else:
            raise ValueError(f'Invalid option for "if_exists": {if_exists}')

    if not fname.parent.exists():
        if create_out_dir:
            fname.parent.mkdir(parents=True)
        else:
            raise IOError(f'Directory "{fname.parent}" does not exist.')

    encoding = {var: dict(zlib=True, complevel=complevel)
                for var in ds.data_vars} if zlib else None

    PBar = {
        True: ProgressBar,
        False: none_context
    }[verbose]

    with PBar(), tempfile.TemporaryDirectory(dir=tmpdir) as tmp:

        fname_tmp = Path(tmp)/fname.name

        if verbose:
            print('Writing:', fname)
            print('Using temporary file:', fname_tmp)

        ds.to_netcdf(path=fname_tmp,
                     engine=engine,
                     encoding=encoding,
                     **kwargs)

        # use intermediary move
        # (both files may be on different devices)
        shutil.move(fname_tmp, str(fname)+'.tmp')
        shutil.move(str(fname)+'.tmp', fname)

    return fname

def none_context(a=None):
    """
    Returns a context manager that does nothing.

    In python 3.7, this is equivalent to `contextlib.nullcontext`.
    """
    return contextmanager(lambda: (x for x in [a]))()



def save_raster(raster: xr.DataArray, 
                outpath: str | Path, 
                extension: str = None, 
                dtype: type = None,
                profile: dict = {}):
    # Check if raster dims are correct
    size = raster.shape
    length = len(size)
    assert length in (2,3), "Only 2D/3D arrays are currently supported"
    outpath = Path(outpath)
    to_check = ['count', 'height', 'width'][-length:]
    for key, cursor in zip(to_check, np.arange(length)):
        assert profile[key] == size[cursor]

    # Check extension
    possible_format = ['tif','nc','png','jpg']
    suffix = outpath.suffix[1:]
    if len(suffix) != 0:
        extension = suffix
    assert extension in possible_format
    
    profile['dtype'] = dtype
    if extension == 'nc':
        save_netcdf(raster, outpath, length)
    else:
        save_rio(raster, outpath, profile, length)

def save_netcdf(raster: xr.DataArray, 
                outpath: str | Path,
                len_dim: int):
    # Reorder raster for to_netcdf function
    if len_dim == 2: 
        raster = xr.Dataset({'output' : raster})
    else:
        dim = set(raster.dims)
        head = dim.difference(['rows','columns','x','y'])
        assert len(head) == 1
        head = head.pop()
        varnames = list(raster.coords[head].values)
        raster = xr.Dataset({name : raster.sel(**{head:name}) for name in varnames})
        raster = raster.drop(head)
    raster = xr.where(raster.isnull(), 0, raster)

    # Write output data
    raster.to_netcdf(outpath)

def save_rio(raster: xr.DataArray, 
             outpath: str | Path,
             profile: dict,
             len_dim: int):
    # Reorder raster for to_raster function
    if 'rows' in list(raster.dims) and 'columns' in list(raster.dims):
        raster = raster.rename({'rows':'y','columns':'x'})
    if len_dim == 2:
        raster = raster.transpose('y','x')
    if len_dim == 3:
        raster = raster.transpose(...,'y','x')

    # Supplement raster with profile tags
    if profile is not None:
        for key in ['nodata', 'transform', 'crs', 'count', 'width', 'height']:
            if key in profile:
                raster.attrs[key] = profile[key]

    # Write output data
    raster.rio.to_raster(outpath, dtype=profile['dtype'], compress='LZW')
    

def save2RGB(raster: xr.DataArray, 
             outpath: str | Path, 
             extension: str = None, 
             band_RGB: list = None,
             dtype: type = None,
             profile: dict = None):
    if band_RGB is None:
        rgb_raster = raster.isel([1,2,3])
    else:
        if check_sel_dict(band_RGB):
            rgb_raster = raster.sel(bands=band_RGB)
        else:
            rgb_raster = raster.isel(bands=band_RGB)
    profile.update({'count':3})
    save_raster(rgb_raster, outpath, extension, dtype, profile)


def save_mask(mask:xr.DataArray, 
              outpath:str | Path, 
              extension:str = None, 
              dtype:type = None,
              profile:dict = None):
    save_raster(mask, outpath, extension, dtype, profile)

def check_sel_dict(input_list:list):
    test = [v > 50 for v in input_list]
    assert np.min(test) == np.max(test)
    return bool(np.max(test))