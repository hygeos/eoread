from dask import array as da
from pathlib import Path
import xarray as xr

from core.geo.naming import names
from eoread.tools import filter_metadata, format_chunks
from core.tools import drop_unused_dims
from core import env, log


def Level1_HYPSO(
        filepath: str|Path,
        chunks: int|tuple = 500,
        metadata_template: list = None,
        v1_compat: bool = False,
        verbose: bool = True,
    ) -> xr.Dataset:
    """
    Read an NTNU HYPSO-1 Level1 product as an xarray.Dataset.
    
    HYPSO-1 is a hyperspectral imaging satellite operated by the Norwegian University
    of Science and Technology (NTNU). It provides high-resolution hyperspectral data
    for ocean monitoring and research.
    
    The dataset contains TOA radiances, viewing/solar angles on the full grid,
    and geolocation information.

    Parameters
    ----------
    filepath : Path or str
        Path to the HYPSO HDF5 file (.h5).
    chunks : int or tuple, optional
        Size of chunks for spatial dimensions. If int, applies to both dimensions.
        If tuple, should be (rows_chunk, columns_chunk).
    metadata_template : list, optional
        List of metadata keys to include. If None, includes all metadata.
        Use empty list [] for minimal metadata.
    verbose : bool, optional
        If True, prints debug messages during reading.
        
    Returns
    -------
    xr.Dataset
        Dataset containing:
            - Lt: Top-of-atmosphere radiance (W/sr/m^2)
            - VZA, VAA, SZA, SAA: Viewing and solar geometry angles
            - lat, lon: Geolocation arrays
            - central_wavelength: Band wavelengths
            - Metadata attributes
            
    Raises
    ------
    AssertionError
        If the file does not exist.
        
    Examples
    --------
    >>> ds = Level1_HYPSO('hypso_product.h5', chunks=1000)
    """
    
    ds = xr.Dataset()
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    
    # Format chunks
    chunks = format_chunks(chunks)

    ds_root = xr.open_datatree(filepath, engine='h5netcdf')
    ds_products = ds_root["products"].to_dataset()
    ds_nav = ds_root["navigation"].to_dataset()

    # get _indirect geographical coordinates and angles if available
    if verbose: log.debug('Read and compute geometric angles')
    ds[str(names.lat)] = ds_nav["latitude"].chunk(chunks)
    ds[str(names.lon)] = ds_nav["longitude"].chunk(chunks)
    ds[str(names.vza)] = ds_nav["sensor_zenith"].chunk(chunks)
    ds[str(names.sza)] = ds_nav["solar_zenith"].chunk(chunks)
    ds[str(names.vaa)] = ds_nav["sensor_azimuth"].chunk(chunks)
    ds[str(names.saa)] = ds_nav["solar_azimuth"].chunk(chunks)

    if verbose: log.debug('Read top of atmosphere data')
    ds[str(names.ltoa)] = ds_products['Lt'].chunk(list(chunks)+[1])
    ds = ds.rename(lines=str(names.rows), samples=str(names.columns), bands=str(names.bands))
    ds[str(names.ltoa)].attrs['unit'] = 'W/sr/m^2'
    
    if verbose: log.debug('Extract central wavelength')
    ds = ds.assign_coords({
        str(names.bands): ds[str(names.bands)].data.astype(str),
    })
    ds = ds.assign({str(names.cwav): ((str(names.bands)), ds_products['Lt'].wavelengths)})

    # Add attributes
    if verbose: log.debug('Add important attributes')
    ds.attrs[str(names.sensor)] = ds_root.attrs['instrument']
    ds.attrs[str(names.platform)] = 'HYPSO'
    ds.attrs[str(names.resolution)] = 40
    ds.attrs[str(names.product_name)] = filepath.name
    ds.attrs[str(names.input_directory)] = str(filepath.parent)
    ds.attrs[str(names.datetime)] = ds_root.attrs['date_aquired']
    
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata'] = filter_fn(ds_root.attrs, metadata_template)

    # ds[naming.flags] = xr.zeros_like(ds.vza, dtype=naming.flags_dtype)
    
    if v1_compat: return _v1_compat(ds)
    return drop_unused_dims(ds).unify_chunks()


def get_sample(level: int=1) -> Path:
    """
    Retrieve a sample HYPSO-1 product file for testing.
    
    Returns path to a pre-configured HYPSO sample product from environment variables.

    Parameters
    ----------
    level : int, optional
        Processing level of the product (currently only level=1 is supported).
        Default is 1.
        
    Returns
    -------
    Path
        Path to the HYPSO HDF5 file.
        
    Raises
    ------
    AssertionError
        If the sample directory does not exist.
        
    Examples
    --------
    >>> hypso_file = get_sample(level=1)
    >>> ds = Level1_HYPSO(hypso_file)
    """
    sample = env.getdir('DIR_SAMPLE_HYPSO')
    assert sample.exists()
    return sample

def _v1_compat(ds):
    
    # Add flags
    ds["flags"] = xr.zeros_like(ds.vza, dtype="uint8")
    
    return ds