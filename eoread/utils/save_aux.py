import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from imageio.v2 import get_writer, imread
from pathlib import Path
from rasterio.transform import Affine
from tempfile import TemporaryDirectory
from itertools import product


########################################################################################
# THIS MODULE WILL BE DEPRECATED, PLEASE DO NOT UPDATE THIS CODE AS IT WILL BE DELETED #
########################################################################################


def to_tif(ds: xr.Dataset, *,
           filename: str | Path = None,
           nodata: int | float = None,
           raster: str = None,
           dtype: str = None,
           compressor: bool = True,
           verbose: bool = True):
            
    # Extract LatLon from dataset
    position = {}
    if 'latitude' in ds and 'longitude' in ds: 
        position['lat'], position['lon'] = ds['latitude'], ds['longitude']
    if 'profile' in ds.attrs:
        position['profile'] = ds.profile
    
    # Standardize Dataset 
    if raster:
        assert raster in ds, f'{raster} variable is missing in dataset'
        ds = ds[raster]
        shape = ds.shape
        # Check data format
        if 'int' not in str(ds.dtype) and verbose: 
            print(f"""[Warning] current data type could be incompatible with \
QGIS colormap, got {ds.dtype}. You should cast your data into integer format""")
    else: 
        ds = _format_dataset(ds)
        shape = ds['latitude'].shape
    assert len(shape) in [2,3], \
        f'Should input 3D or 2D dataset, not {len(ds.shape)}D'
        
    # Generate profile
    ds.attrs = {}
    if 'profile' in position:
        ds.attrs.update(position['profile'])
    else:
        ds.attrs['count']  = shape[-3] if len(shape) == 3 else 1
        ds.attrs['width']  = shape[-2]
        ds.attrs['height'] = shape[-1]
        ds.attrs['crs']    = '+proj=latlong'
        if nodata: ds.attrs['nodata'] = nodata
        if compressor: ds.attrs['compress'] = 'lzw'
        
        if 'lat' in position:
            ds.attrs['transform'] = _get_transform(position['lat'],position['lon'])
    
    # Rename dimension into xy
    assert 'x' in ds.dims and 'y' in ds.dims
    if len(ds.dims) == 2: ds = ds.transpose('y','x')
    if len(ds.dims) == 3: ds = ds.transpose(...,'y','x')
    
    return ds.rio.to_raster(raster_path=filename, dtype=dtype, 
                            recalc_transform=False, compute=True)

def to_img(ds: xr.Dataset | xr.DataArray = None,
           array: np.ndarray = None, *,
           filename: str | Path = None,
           vmin: float = None,
           vmax: float = None,
           rgb: list = None,
           raster: str = None,
           cmap: str = 'viridis',
           verbose: bool = True):
    """
    Function to save array into image file format

    Args:
        ds (xr.Dataset | xr.DataArray, optional): Input data
        array (np.ndarray, optional): Input array
        filename (str | Path, optional): Output file path. Defaults to None.
        vmin (float, optional): Minimum value for colorbar. Defaults to None.
        vmax (float, optional): Maximum value for colorbar. Defaults to None.
        rgb (list, optional): Bands to select as RGB. Defaults to None.
        raster (str, optional): Name of the variable to save. Defaults to None.
        cmap (str, optional): Colormap to use for mask. Defaults to 'viridis'.
        compressor (str, optional): Option to save into compressed format. Defaults to None.
        verbose (bool, optional): Option for logging. Defaults to True.

    Returns:
        str: Output file path
    """
    assert (ds is not None) ^ (array is not None), 'Please fill only `ds` or `array`, not both'
    assert isinstance(array, np.ndarray) or (array is None), \
        f'Wrong input array format, got {type(array)}'
    assert isinstance(ds, (xr.Dataset, xr.DataArray)) or (ds is None), \
        f'Wrong input data format, got {type(ds)}'
    
    # Extract DataArray from Dataset
    if isinstance(ds, xr.Dataset):
        assert raster is not None, f'Please enter raster attribute'
        ds = ds[raster]
        
    # Initialize min and max values for colorbar
    if array is not None:
        if vmin is None: vmin = np.min(array)
        if vmax is None: vmax = np.max(array)
        assert len(array.shape) == 2, 'xr.DataArray is not 2D'
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        image = cmap(norm(array))
        plt.imsave(filename, image)
        return filename
    else:
        if vmin is None: vmin = ds.min().values
        if vmax is None: vmax = ds.max().values    
    
    # Manage save of RGB image and mask
    if rgb:
        assert len(rgb) == 3
        assert len(ds.shape) == 3, 'xr.DataArray is not 3D'
        assert all(i in ds[ds.dims[0]] for i in rgb)
        ds = ds.sel({ds.dims[0]: rgb})
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        image = norm(ds.transpose('y','x',...).values)
    else:
        assert len(ds.shape) == 2, 'xr.DataArray is not 2D'
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        image = cmap(norm(ds.values))
    
    plt.imsave(filename, image)
    return filename

def to_gif(ds: xr.Dataset | xr.DataArray, *,
           filename: str | Path = None,
           vmin: float = None,
           vmax: float = None,
           rgb: list = None,
           raster: str = None,
           cmap: str = 'viridis',
           time_dim: str = None,
           duration: int = 1,
           loop: bool = True,
           verbose: bool = True):
    """
    Function to save 3D or 4D array into animated format (GIF)

    Args:
        ds (xr.Dataset | xr.DataArray): Input
        filename (str | Path, optional): Output file path. Defaults to None.
        vmin (float, optional): Minimum value for colorbar. Defaults to None.
        vmax (float, optional): Maximum value for colorbar. Defaults to None.
        rgb (list, optional): Bands to select as RGB. Defaults to None.
        raster (str, optional): Name of the variable to save. Defaults to None.
        cmap (str, optional): Colormap to use for mask. Defaults to 'viridis'.
        time_dim (str, optional): Dimension to use as time. Defaults to None.
        duration (int, optional): Time interval between each frame in seconds. Defaults to 1.
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
    
    # Create GIF file
    gif = GifMaker(gif_file=filename, duration=duration, loop=loop)
    with TemporaryDirectory() as tmpdir:
        for i,_ in enumerate(ds[time_dim]):
            outpath = Path(tmpdir)/f'img_time_{i}.png'
            to_img(ds.isel({time_dim:i}), filename=outpath, verbose=verbose,
                   rgb=rgb, cmap=cmap, vmin=vmin, vmax=vmax)
            gif.add_image(filename=outpath)
    gif.write()
    
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
    lonf, latf = lon.values, lat.values
    a = (lon[0,-1].values-lon[0,0].values)/size[1]
    b = 0 #(lon[-1,0].values-lon[0,0].values)/size[1]
    c = lon[0,0].values
    d = 0 #(lat[0,-1].values-lat[0,0].values)/size[0]
    e = (lat[-1,0].values-lat[0,0].values)/size[0]
    f = lat[0,0].values    
    return Affine(a,b,c,d,e,f)


class GifMaker:
    
    def __init__(self, 
                 gif_file: str | Path = None, 
                 duration: float = 1, 
                 loop: bool = True,
                 tmpdir: str | Path = None) -> None:
        """
        A class to facilitate the creation of gif files
        
        Args:
            gif_file (str | Path, optional): Path to store the GIF file. Defaults to None.
            duration (float, optional): Time interval between each frame in seconds. Defaults to 1.
            tmpdir (str | Path, optional): Path of temporary directory to use. Defaults to None.
        """
        self.gif_file = gif_file
        self.duration = duration
        self.tmpdir   = tmpdir
        self.current  = get_writer(gif_file, mode='I', duration=duration, loop=loop)
    
    def __del__(self):
        self.write()
    
    def add_image(self, filename: str | Path = None, arr: np.ndarray = None):
        assert (filename is not None) ^ (arr is not None)
        if arr:
            with TemporaryDirectory(dir=self.tmpdir) as tmpdir:
                filename = Path(tmpdir)/'frame.png'
                plt.imsave(filename, arr)
                self.current.append_data(imread(filename)) 
        if filename: self.current.append_data(imread(filename))

    def savefig(self, **kwargs):
        with TemporaryDirectory(dir=self.tmpdir) as tmpdir:
            img_file = Path(tmpdir)/'frame.png'
            plt.savefig(img_file, **kwargs)
            self.add_image(img_file)

    def write(self):
        self.current.close()