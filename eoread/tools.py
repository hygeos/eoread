from core.interpolate import interp, Linear, Nearest
from core.import_utils import import_module
from core.uncompress import uncompress
from core.geo.naming import names
from core.tools import only
from core import env

from typing import Union, Literal
from xarray import DataArray, Dataset, open_dataarray
from dask.array import meshgrid, linspace
from tempfile import TemporaryDirectory
from pathlib import Path


def format_chunks(chunks: Union[int, list, tuple, dict]) -> dict:
    """
    Normalize chunk specification to a dictionary format.
    
    Converts various chunk size specifications (int, list, tuple) to a
    standardized dictionary with 'rows' and 'columns' keys.
    
    Parameters
    ----------
    chunks : int, list, tuple, or dict
        Chunk size specification:
        - int: Same chunk size for both dimensions
        - list/tuple: [rows_chunk, columns_chunk]
        - dict: Already in correct format
        
    Returns
    -------
    dict
        Dictionary with 'rows' and 'columns' chunk sizes.
        
    Raises
    ------
    AssertionError
        If chunks format is invalid.
        
    Examples
    --------
    >>> format_chunks(500)
    {'rows': 500, 'columns': 500}
    >>> format_chunks([1000, 500])
    {'rows': 1000, 'columns': 500}
    """
    
    # Manage different chunk types
    if isinstance(chunks, int):
        chunks = [chunks, chunks]
    if isinstance(chunks, list|tuple):
        assert len(chunks) == 2
        chunks = {str(names.rows): chunks[0], str(names.columns): chunks[1]}
    
    assert len(chunks) == 2, 'chunks should be for spatial dimensions'
    assert str(names.rows) in chunks and str(names.columns) in chunks
    return chunks


def filter_metadata(metadata: dict, template: list) -> dict:
    """
    Short method to filter metadata dictionary based on a template

    Parameters
    ----------
    metadata : dict
        Dictionary corresponding to metadata.
    template : list
        List of list describing nodes to keep.
    
    Examples
    --------
    >> d = {'a': 0, 'b': {'c': 1, 'd':2}}
    >> t = [['a'], ['b','c']]
    >> filter_metadata(d, t)
    {'a': 0, 'b': {'c': 1}}
    """  
    
    # Populate new dictionary with nodes to kept 
    result = {}    
    for key_list in template:
        current_meta = metadata
        current_result = result
        
        # Traverse the nested dictionary using the provided keys
        for key in key_list[:-1]:
            if isinstance(current_meta, dict) and key in current_meta:
                current_result[key] = {}
                current_result = current_result[key]
                current_meta = current_meta[key]
            else: 
                raise ValueError(f'{key} not found')
            
        # If the last key is present, add it to the result
        if isinstance(current_meta, dict) and key_list[-1] in current_meta:
            current_result.update({key_list[-1]: current_meta[key_list[-1]]})
        else:
            raise ValueError(f'Leaf {key_list[-1]} has been found')
    
    return result


def spatial_resample(
        array: DataArray,
        output_shape: dict, 
        chunks: dict,
        method: Literal['linear','repeat'] = 'linear'
    ) -> DataArray:
    """
        ratio: list[x_ratio, y_ratio]
        if ratio is > 1 then downsamples
        if ratio is < 1 then upscales
    """
    
    # Check inputs compliance
    array = array.squeeze()
    assert len(array.shape) == 2, 'Array should be 2D'
    assert all(k in array.dims for k in output_shape.keys())
    assert chunks.keys() == output_shape.keys()
    
    # Compute the ratio of transformation
    dims = array.dims
    ratio = {d: array.sizes[d]/output_shape[d] for d in dims}
    
    # No resolution change => return the input DataArray
    if all(v == 1 for v in ratio.values()):
        return array
    
    # downsample
    if any(v > 1 for v in ratio.values()):
        assert all(r == int(r) for r in ratio.values())
        params = {d: int(ratio[d]) for d in dims}
        resampled = array.coarsen(**params, boundary="trim")
        resampled = resampled.mean().chunk(chunks)
        
    # over-sample
    else:
        # Interpolator requires numpy-backed data; compute the (small) tie-point array
        array = array.compute(scheduler='sync')
        coords = array.coords
        method = Linear if method == 'linear' else Nearest
        xy = meshgrid(*[
            linspace(coords[d].values[0], coords[d].values[-1], output_shape[d]) 
            for d in dims
        ])
        params = {d: method(DataArray(xy[i], dims=('d1','d0'))) for i,d in enumerate(dims)}
        resampled = interp(array, **params)
        resampled = resampled.rename(d0=dims[0], d1=dims[1])
        
        # new = {d: np.linspace(
        #     coords[d].values[0], coords[d].values[-1], output_shape[d]
        # ) for d in dims}
        # params = {d: method(DataArray(new[d], dims=(d))) for d in dims}
        # resampled = interp(array, **params)
    
    return resampled


def open_raster(
        dirname: Union[str, Path], 
        pattern: str, 
        compress_ext: str = None, 
        engine='h5netcdf'
    ) -> DataArray:
    """
    Find and open a raster file matching a pattern.
    
    Searches for a single file matching the pattern and optionally
    decompresses it before opening.
    
    Parameters
    ----------
    dirname : str or Path
        Directory to search in.
    pattern : str
        Glob pattern to match (e.g., '*_B01.jp2').
    compress_ext : str, optional
        If provided, uncompresses file with this extension first.
        Default is None.
    engine : str, optional
        xarray engine for opening the file. Default is 'h5netcdf'.
        
    Returns
    -------
    DataArray
        DataArray with the raster data (squeezed to remove size-1 dimensions).
        
    Raises
    ------
    AssertionError
        If pattern matches zero or multiple files.
        
    Examples
    --------
    >>> arr = open_raster('/data/', '*_B02.tif', engine='rasterio')
    >>> arr = open_raster('/data/', '*_CLD.zip', compress_ext='.zip')
    """
    # Find file path
    path = only(list(Path(dirname).glob(pattern)))
    
    # Open and uncompress if needed 
    if compress_ext: 
        with TemporaryDirectory() as tmpdir:
            path = uncompress(path, tmpdir)
            return open_dataarray(path, engine=engine).squeeze()
    
    return open_dataarray(path, engine=engine).squeeze()


def collect_sample(
        variable: str,
        provider: Literal[None, 'cnes'],
        sand_collection: str|None = None,
        level: int|None = None
    ) -> Path:
    
    # Check if user has provided a path
    variable = env.getvar(variable, default='')  
    
    # If not provided, try to download a sample with SAND
    if variable == '':
        
        if provider is not None:
            assert sand_collection and level, 'Provide kwargs for SAND downloader'
        else: 
            raise ValueError('')
        
        # Check SAND importation
        try: 
            from sand.sample_product import products
        except ImportError:
            raise ImportError('To use get_sample function, you need to install SAND module')
        
        # Collect appropriated provider
        downloader = {
            'cdse': 'sand.copernicus_dataspace.DownloadCDSE',
            'cnes': 'sand.cnes.DownloadCNES',
            'eumdac': 'sand.eumdac.DownloadEumDAC',
            'nasa': 'sand.nasa.DownloadNASA',
            'usgs': 'sand.usgs.DownloadUSGS',
        }[provider]
        
        # Retrieve name of example product
        prod_id = products[sand_collection][f'l{level}_product']
        
        # Download product with SAND
        dl = import_module(downloader)()
        directory = env.getdir('DIR_SAMPLES')/sand_collection
        target = dl.download_file(prod_id, directory)
        
        assert target.exists()
        return target
        
    else:
        return Path(variable)
    

def crop(
        ds: Dataset, 
        latmin: float|None = None, 
        latmax: float|None = None,
        lonmin: float|None = None, 
        lonmax: float|None = None,
        drop: bool = True
    ) -> Dataset:
    """
    Crop output of eoread reader based on latitude and longitude arrays.

    Parameters
    ----------
    ds : xr.Dataset
        Output from an eoread reader.
    latmin : float or None, optional
        Minimum of latitude. Default is None.
    latmax : float or None, optional
        Maximum of latitude. Default is None.
    lonmin : float or None, optional
        Minimum of longitude. Default is None.
    lonmax : float or None, optional
        Maximum of longitude. Default is None.
    drop : bool, optional
        Option to drop invalid pixels. If False, invalid pixels are set to NaN. Default is True.
    """
    # Filter latitude and longitude
    latmask = (ds[str(names.lat)] >= latmin) & (ds[str(names.lat)] <= latmax)
    lonmask = (ds[str(names.lon)] >= lonmin) & (ds[str(names.lon)] <= lonmax)
    
    # Load masks to determine output shape 
    if drop: 
        latmask = latmask.compute()
        lonmask = lonmask.compute()
        
    return ds.where(latmask & lonmask, drop=drop)