#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for saving results
'''

import shutil
import tempfile
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from typing import Union
from pathlib import Path
from contextlib import contextmanager
from dask.diagnostics import ProgressBar
from rasterio.transform import Affine
from tempfile import TemporaryDirectory
from itertools import product

from .naming import naming


def to_netcdf(ds: xr.Dataset, *,
              filename: Union[str, Path] = None,
              dirname: Union[str, Path] = None,
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
    """
    Write an xarray Dataset `ds` using `.to_netcdf` with several additional features:
    - construct file name using  `dirname`, `product_name` and `ext`
      (optionnally - otherwise with `filename`)
    - check that the output file does not exist already
    - Use file compression
    - Use temporary file
    - Create output directory if it does not exist

    Args:
        ds (xr.Dataset): _description_
        filename (Union[str, Path], optional): Output file path, if None, build filename from `dirname`, `product_name` and `ext`. Defaults to None.
        dirname (Union[str, Path], optional): directory for output file (default None: uses the attribute input_directory of ds). Defaults to None.
        product_name (str, optional): base name for the output file. If None (default), use the attribute named attr_name. Defaults to None.
        ext (str, optional): extension. Defaults to '.nc'.
        product_name_attr (str, optional): name of the attribute to use for product_name in `ds`. Defaults to 'product_name'.
        if_exists (str, optional): what to do if output file exists. Defaults to 'error'.
        tmpdir (str, optional): use a given temporary directory. Defaults to None.
        create_out_dir (str, optional): create output directory if it does not exist. Defaults to True.
        engine (str, optional): Engine driver to use. Defaults to 'h5netcdf'.
        zlib (bool, optional): _description_. Defaults to True.
        complevel (int, optional): Compression level. Defaults to 5.
        verbose (bool, optional): _description_. Defaults to True.

    Raises:
        IOError: _description_
        ValueError: _description_
        IOError: _description_

    Returns:
        str: output file name
    """
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


def to_tif(ds: xr.Dataset, *,
           filename: str | Path = None,
           nodata: int | float = None,
           raster: str = None,
           compressor: bool = True,
           verbose: bool = True):
            
    # Extract LatLon from dataset
    assert 'latitude' in ds, 'Latitude variable is missing'
    assert 'longitude' in ds, 'Longitude variable is missing'
    lat, lon = ds['latitude'], ds['longitude']
    
    # Standardize Dataset 
    if raster:
        assert raster in ds, f'{raster} variable is missing in dataset'
        ds = ds[raster]
        shape = ds.shape
        # Check data format
        if 'int' not in str(ds.dtype): 
            print(f"""[Warning] current data type could be incompatible with 
QGIS colormap, got {ds.dtype}. You should cast your data into integer format""")
    else: 
        ds = _format_dataset(ds)
        shape = ds['latitude'].shape
    assert len(shape) in [2,3], \
        f'Should input 3D or 2D dataset, not {len(ds.shape)}D'
        
    # Generate profile
    ds.attrs['count']  = shape[-3] if len(shape) == 3 else 1
    ds.attrs['width']  = shape[-2]
    ds.attrs['height'] = shape[-1]
    ds.attrs['crs']    = '+proj=latlong'
    if nodata: ds.attrs['nodata'] = nodata
    if compressor: ds.attrs['compress'] = 'lzw'
    ds.attrs['transform'] = _get_transform(lat,lon)
    
    # Rename dimension into xy
    assert 'x' in ds.dims and 'y' in ds.dims
    if len(ds.dims) == 2: ds = ds.transpose('y','x')
    if len(ds.dims) == 3: ds = ds.transpose(...,'y','x')
    
    return ds.rio.to_raster(raster_path=filename, recalc_transform=False, compute=True)

def to_img(ds: xr.Dataset | xr.DataArray, *,
           filename: str | Path = None,
           rgb: list = None,
           raster: str = None,
           cmap: str = 'viridis',
           compressor: str = None,
           verbose: bool = True):
    """
    Function to save array into image file format

    Args:
        ds (xr.Dataset | xr.DataArray): Input data
        filename (str | Path, optional): Output file path. Defaults to None.
        rgb (list, optional): Bands to select as RGB. Defaults to None.
        raster (str, optional): Name of the variable to save. Defaults to None.
        cmap (str, optional): Colormap to use for mask. Defaults to 'viridis'.
        compressor (str, optional): Option to save into compressed format. Defaults to None.
        verbose (bool, optional): Option for logging. Defaults to True.

    Returns:
        str: Output file path
    """
    
    assert isinstance(ds, (xr.Dataset, xr.DataArray)), \
        f'Wrong input data format, got {type(ds)}'
    
    # Extract DataArray from Dataset
    if isinstance(ds, xr.Dataset):
        assert raster is not None, f'Please enter raster attribute'
        ds = ds[raster]
    
    # Manage save of RGB image and mask
    if rgb:
        assert len(rgb) == 3
        assert len(ds.shape) == 3, 'xr.DataArray is not 3D'
        assert all(i in ds[ds.dims[0]] for i in rgb)
        ds = ds.sel({ds.dims[0]: rgb})
        norm = plt.Normalize(vmin=ds.min().values, vmax=ds.max().values)
        image = norm(ds.transpose('y','x',...).values)
    else:
        assert len(ds.shape) == 2, 'xr.DataArray is not 2D'
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=ds.min().values, vmax=ds.max().values)
        image = cmap(norm(ds.values))
    
    plt.imsave(filename, image)
    return filename

def to_gif(ds: xr.Dataset | xr.DataArray, *,
           filename: str | Path = None,
           rgb: list = None,
           raster: str = None,
           cmap: str = 'viridis',
           time_dim: str = None,
           duration: int = 40,
           compressor: str = None,
           verbose: bool = True):
    """
    Function to save 3D or 4D array into animated format (GIF)

    Args:
        ds (xr.Dataset | xr.DataArray): Input
        filename (str | Path, optional): Output file path. Defaults to None.
        rgb (list, optional): Bands to select as RGB. Defaults to None.
        raster (str, optional): Name of the variable to save. Defaults to None.
        cmap (str, optional): Colormap to use for mask. Defaults to 'viridis'.
        time_dim (str, optional): Dimension to use as time. Defaults to None.
        duration (int, optional): Time interval between each frame in milliseconds. Defaults to 40.
        compressor (str, optional): Option to save into compressed format. Defaults to None.
        verbose (bool, optional): Option for logging. Defaults to True.

    Returns:
        str: Output file path
    """
    
    assert isinstance(ds, (xr.Dataset, xr.DataArray)), \
        f'Wrong input data format, got {type(ds)}'
    
    # Extract DataArray from Dataset
    if isinstance(ds, xr.Dataset):
        assert raster is not None, f'Please enter raster attribute'
        ds = ds[raster]
    assert time_dim in ds.dims
    
    list_img = []
    with TemporaryDirectory() as tmpdir:
        for i,_ in enumerate(ds[time_dim]):
            outpath = Path(tmpdir)/f'img_time_{i}.png'
            to_img(ds.isel({time_dim:i}), filename=outpath, verbose=verbose,
                   rgb=rgb,cmap=cmap, compressor=compressor)
            list_img.append(Image.open(outpath))
    
    list_img[0].save(filename, save_all=True, append_images=list_img[1:], 
                     optimize=False, duration=duration, loop=0)
    
    return filename


def _format_dataset(ds : xr.Dataset):
    ds_dims = list(ds.dims)
    assert 'x' in ds_dims and 'y' in ds_dims
    for var in ds:
        dims = list(ds[var].dims)
        if 'x' not in dims and 'y' not in dims: continue
        dims.remove('x')
        dims.remove('y')
        if len(dims) == 0: continue
        items = product(*(ds[var][d].values for d in dims))
        for item in items:
            new_dims = {dims[i]:item[i] for i in range(len(item))}
            varname = var
            for k,v in new_dims.items(): varname += f'_{k}_{v}'
            ds[varname] = ds[var].sel(new_dims)
    ds_dims.remove('x')
    ds_dims.remove('y')
    ds = ds.drop_dims(ds_dims)
    return ds

def _get_transform(lat: xr.DataArray, lon: xr.DataArray):
    # Compute transformation coefficient
    size = lat.shape
    a = (lon[0,-1]-lon[0,0])/size[1]
    b = (lon[-1,0]-lon[0,0])/size[1]
    c = lon[0,0]
    d = (lat[0,-1]-lat[0,0])/size[0]
    e = (lat[-1,0]-lat[0,0])/size[0]
    f = lat[0,0]    
    return Affine(a,b,c,d,e,f)


class GifMaker:
    """
    A class to facilitate the creation of gif files
    """
    def __init__(self, gif_file=None, duration=1, tmpdir=None) -> None:
        self.gif_file = gif_file
        self.duration = duration
        self.tmpdir = tmpdir
        self.list_images = []
    
    def add_image(self, filename):
        self.list_images.append(imageio.imread(filename))

    def savefig(self, **kwargs):
        with TemporaryDirectory(dir=self.tmpdir) as tmpdir:
            img_file = Path(tmpdir)/'frame.png'
            plt.savefig(img_file, **kwargs)
            self.add_image(img_file)

    def write(self, gif_file=None):
        assert (gif_file is None) != (self.gif_file is None), \
            'gif_file must be provided (and only once)'
        gif_file = gif_file or self.gif_file
        
        imageio.mimsave(
            gif_file,
            self.list_images,
            duration=self.duration)
    
    def display(self):
        """
        Display the gif in a notebook
        """
        with TemporaryDirectory(dir=self.tmpdir) as tmpdir:
            file_gif = Path(tmpdir)/'file.gif'
            self.write(file_gif)
            display(Image(filename=file_gif))