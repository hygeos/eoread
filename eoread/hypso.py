from dask import array as da
from pathlib import Path
import xarray as xr

from eoread.utils import filter_metadata
from eotools.solar_irradiance import solar_irradiance
from core.interpolate import interp, Linear
from core.geo import n
from core.tools import  drop_unused_dims
from core import env, log



#rtoa  = Var("Rtoa", attrs={'desc': "Top of Atmosphere reflectance"}, dtype='float32')





def Level1_HYPSO(filepath: str|Path,
                 chunks: int|tuple = 500,
                 metadata_template: list = None,
                 v1_compat: bool = True) -> xr.Dataset:
    '''
    Read an NTNU HYPSO Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, 
    the angles on the full grid, etc.

    Arguments:
        filepath: Path of the HYPSO file
        chunks: Size of chunks for spatial axis
        metadata_template: If None, add all metadata in output xarray.Dataset attributes else add only specified metadata.
        v1_compat: Option to format output xarray.Dataset such as version 1
    '''
    
    ds = xr.Dataset()
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'


    ds_root = xr.open_datatree(filepath, engine='h5netcdf')

    match str(ds_root.attrs['processing_level']).upper():

        case "L1C":
            hypso_product_name = "Lt"
            polymer_product_name = n.ltoa
            units = 'W/sr/m^2'
            load_f0 = True
            
        case "L1D":
            hypso_product_name = "Rhot"
            polymer_product_name = n.rtoa
            units = ''
            load_f0 = False

        case _:
            log.debug('Unsupported HYPSO product level.')
            return

    
    print(ds_root['metadata']['corrections'].attrs['radiometric_coefficents_version'])


    if isinstance(chunks, int): chunks = [chunks]*2
    ds_products = ds_root["products"].to_dataset()
    try:
        ds_nav = ds_root["navigation"].to_dataset()
    except KeyError:
        ds_nav = ds_root["geometry"].to_dataset()

    # get _indirect geographical coordinates and angles if available
    log.debug('Read and compute geometric angles')
    ds[str(n.lat)] = ds_nav["latitude"].chunk(chunks)
    ds[str(n.lon)] = ds_nav["longitude"].chunk(chunks)
    ds[str(n.vza)] = ds_nav["sensor_zenith"].chunk(chunks)
    ds[str(n.sza)] = ds_nav["solar_zenith"].chunk(chunks)
    ds[str(n.vaa)] = ds_nav["sensor_azimuth"].chunk(chunks)
    ds[str(n.saa)] = ds_nav["solar_azimuth"].chunk(chunks)

    log.debug('Read top of atmosphere data')
    try:
        ds[str(polymer_product_name)] = ds_products[hypso_product_name].chunk(list(chunks)+[1])
        ds = ds.rename(lines=str(n.rows), samples=str(n.columns), bands=str(n.bands))
        ds[str(polymer_product_name)].attrs['unit'] = 'W/sr/m^2'
        wavelengths = ds_products[hypso_product_name].wavelengths
    except KeyError:

        import numpy as np

        wavelengths = []

        for band_idx, band_key in enumerate(ds_products.keys()):

            if band_idx == 0:
                rows, columns = ds_products[band_key].shape
                bands = len(ds_products.keys())
                datacube = np.empty((rows,columns,bands))

            band_data = np.array(ds_products[band_key][:], dtype='float')

            wave = ds_products[band_key].attrs["wavelength"]

            wavelengths.append(wave)

            datacube[:,:,band_idx] = band_data


        ds[str(polymer_product_name)] = xr.DataArray(datacube, name=hypso_product_name, dims=['lines', 'samples', 'bands']).chunk(list(chunks)+[1])
        ds = ds.rename(lines=str(n.rows), samples=str(n.columns), bands=str(n.bands))
        ds[str(polymer_product_name)].attrs['unit'] = units

        wavelengths = np.array(wavelengths)

        wavelengths = np.around(wavelengths,1) # round to one decimal like hypso-package L1 processing
        wavelengths = np.array(wavelengths).astype(np.int32) # convert to int like hypso-package L1 processing
        wavelengths = np.array(wavelengths, dtype='float32') # convert to float for NetCDF writing compatibility



    log.debug('Extract central wavelength')
    #ds = ds.assign_coords({str(n.bands): da.arange(len(ds[str(n.bands)]))+1})
    ds = ds.assign_coords({str(n.bands): wavelengths})
    
    ds = ds.assign({str(n.cwav): ((str(n.bands)), wavelengths),
         str(n.wav): ((str(n.bands)), wavelengths),
         str(n.bnames): ((str(n.bands)), ds[str(n.bands)].data.astype(str))})

    if load_f0:
        # read solar irradiance
        F0 = solar_irradiance("LISIRD", variant="1nm")

        print(F0)
        interpolated_F0 = interp(F0, wavelength=Linear(wavelengths))
        #ds["F0"] = interp(F0, wavelength=Linear(ds.wav))

        # convert it to a unit compatible with Ltoa
        #assert ds.F0.units == "W m-2 nm-1"
        #ds["F0"] = ds.F0 * 1000
        #ds.F0.attrs.update(units="W m-2 um-1")
        #ds = ds.rename(lines="y", samples="x")


        interpolated_F0 = interpolated_F0*1000
        ds["F0"] = xr.DataArray(interpolated_F0, name="F0")
        #ds.rename(lines=str(n.rows), samples=str(n.columns))
        ds["F0"].attrs.update(units="W m-2 um-1")

    # acquisition datetime
    log.debug('Add important attributes')
    ds.attrs[str(n.sensor)] = ds_root.attrs['instrument']
    ds.attrs[str(n.platform)] = 'HYPSO'
    ds.attrs[str(n.resolution)] = 40
    ds.attrs[str(n.product_name)] = filepath.name
    ds.attrs[str(n.input_directory)] = str(filepath.parent)

    try:
        ds.attrs[str(n.datetime)] = ds_root.attrs['date_aquired']
    except KeyError:

        timestr = ds_root.attrs['timestamp_acquired']
        if timestr.endswith('+00:00Z'):
            timestr = timestr.replace('+00:00Z', 'Z')  # or '+00:00'


        ds.attrs[str(n.datetime)] = timestr



    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata'] = filter_fn(ds_root.attrs, metadata_template)

    ds.attrs.pop("metadata")

    # ds[naming.flags] = xr.zeros_like(ds.vza, dtype=naming.flags_dtype)
    
    if v1_compat: return _v1_compat(ds)
    return drop_unused_dims(ds).unify_chunks()


def get_sample(level: int=1) -> Path:
    """
    Bring a HYPSO file path to test reading function

    Args:
        level (int, optional): Level of the product. Defaults to 1.
        use_cache (bool, optional): Option to save the result of the query to the download API to speed up the process. Defaults to True.
    """
    sample = env.getdir('DIR_SAMPLE_HYPSO')
    assert sample.exists()
    return sample

def _v1_compat(ds):
    
    # Add flags
    ds["flags"] = xr.zeros_like(ds.vza, dtype="uint16")
    
    return ds
