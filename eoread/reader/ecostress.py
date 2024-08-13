from eoread.utils.tools import merge
from eoread.utils.config import load_config
from eoread.utils.naming import naming as n
from eoread.download.download_nextcloud import download_nextcloud
from pathlib import Path
from os.path import exists

import numpy as np
import xarray as xr 
import dask.array as da



bands_tir = [8290,8780,9200,10490,12090]

band_index = {  # Bands      - wavelength (um)   - resolution (m)    - group
    8290:  1,   # Band 1        0.62 - 0.67	         250              250m
    8780:  2,   # Band 2        0.84 - 0.87	         250              250m
    9200:  3,   # Band 3        0.46 - 0.48	         500              500m
    10490: 4,   # Band 4        0.54 - 0.56          500              500m
    12090: 5,   # Band 5   	    1.23 - 1.25	         500              500m
    }


def Level1_ECOSTRESS(filepath: Path | str,
                     radiometry: str = 'reflectance',
                     chunks: int = 500,
                     LUT_file: str = None,
                     split: bool = False):
    # Revize variables
    filepath = Path(filepath)
    try:
        raw = xr.open_dataset(filepath, group='HDFEOS/GRIDS/ECO_L1CG_RAD_70m/Data Fields')
    except ValueError as e: 
        raise ImportError(f"You must install 'h5netcdf' library to use ECOSTRESS reader, got message : {e}")
    raw = raw.chunk(chunks=chunks)
    
    # Read Metadata
    granule_mtd = xr.open_dataset(filepath, group='HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata')
    attributes  = xr.open_dataset(filepath, group='HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata')
    info = str(xr.open_dataset(filepath, group='HDFEOS INFORMATION')['StructMetadata.0'].values)
    
    # Change radiometry of input data 
    if LUT_file:
        assert exists(LUT_file), f'{LUT_file} does not exist'
        LUT_file = xr.open_dataset(LUT_file, group='lut').astype('float32')
    l1 = transform_radiometry(raw, radiometry, split, granule_mtd, LUT_file)   
    
    # Change dimensions name and update coordinates
    if not split:
        new_dims = [n.rows,n.columns,n.bands_tir]
        coords = {n.bands_tir: list(band_index.keys())}
    else: new_dims, coords = [n.rows,n.columns], {}
    
    revize_dims = dict(zip(list(l1.dims), new_dims))
    l1 = l1.rename_dims(revize_dims)
    l1 = l1.assign_coords(coords)
    
    # Summarize Attributes
    to_parse = [attr.split("=") for attr in info.split('\n') if len(attr) != 0]
    info = parse_attrs(to_parse)
    l1.attrs['Description']     = str(attributes.LongName.values) 
    l1.attrs[n.product_name]    = str(attributes.LocalGranuleID.values)[:-3]
    l1.attrs[n.input_directory] = str(filepath.parent)
    l1.attrs[n.datetime]   = str(attributes.ProductionDateTime.values)
    l1.attrs[n.resolution] = 70
    l1.attrs[n.platform]   = str(attributes.PlatformLongName.values)
    l1.attrs[n.sensor]     = str(attributes.InstrumentShortName.values)
    l1.attrs[n.shortname]  = str(attributes.ShortName.values)
    l1.attrs['night']      = str(str(attributes.DayNightFlag.values) != 'Day')
    l1.attrs['CRS']        = str(attributes.CRS.values)
    l1.attrs['Boundary']   = str(attributes.SceneBoundaryLatLonWKT.values)
    l1.attrs['version']    = str(attributes.PGEVersion.values)  
    
    l1 = supplement_latlon(l1, chunks)
    return l1


def Level2_ECOSTRESS(filepath: Path | str,
                     radiometry: str = 'reflectance',
                     chunks: int = 500,
                     split: bool = False):
    # Revize variables
    filepath = Path(filepath)
    try:
        raw = xr.open_dataset(filepath, group='HDFEOS/GRIDS/ECO_L2G_LSTE_70m/Data Fields')
    except ValueError as e: 
        raise ImportError(f"You must install 'h5netcdf' library to use ECOSTRESS reader, got message : {e}")
    l1 = raw.chunk(chunks=chunks)
    
    # Read Metadata
    granule_mtd = xr.open_dataset(filepath, group='HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata')
    attributes  = xr.open_dataset(filepath, group='HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata')
    info = str(xr.open_dataset(filepath, group='HDFEOS INFORMATION')['StructMetadata.0'].values)
    
    # Change dimensions name and update coordinates
    new_dims, coords = [n.rows,n.columns], {}
    
    revize_dims = dict(zip(list(l1.dims), new_dims))
    l1 = l1.rename_dims(revize_dims)
    l1 = l1.assign_coords(coords)
    
    # Summarize Attributes
    to_parse = [attr.split("=") for attr in info.split('\n') if len(attr) != 0]
    info = parse_attrs(to_parse)
    l1.attrs['Description']     = str(attributes.LongName.values) 
    l1.attrs[n.product_name]    = str(attributes.LocalGranuleID.values)[:-3]
    l1.attrs[n.input_directory] = str(filepath.parent)
    l1.attrs[n.datetime]   = str(attributes.ProductionDateTime.values)
    l1.attrs[n.resolution] = 70
    l1.attrs[n.platform]   = str(attributes.PlatformLongName.values)
    l1.attrs[n.sensor]     = str(attributes.InstrumentShortName.values)
    l1.attrs[n.shortname]  = str(attributes.ShortName.values)
    l1.attrs['night']      = str(str(attributes.DayNightFlag.values) != 'Day')
    l1.attrs['CRS']        = str(attributes.CRS.values)
    l1.attrs['Boundary']   = str(attributes.SceneBoundaryLatLonWKT.values)
    l1.attrs['version']    = str(attributes.PGEVersion.values)  
    
    l1 = supplement_latlon(l1, chunks)
    return l1



def transform_radiometry(raw_data, radiometry, split, granule_mtd, LUT_file):
    assert radiometry in ['radiance','reflectance'], \
        f'Invalid radiometry value, get {radiometry}'
    
    quality_vars = [k for k in raw_data.keys() if 'quality' in k]
    level1 = raw_data.drop_vars(quality_vars)
    flags = raw_data.data_quality_1 != 0
    level1[n.flags] = flags.astype(n.flags_dtype)
    
    # Process Emissive bands
    if radiometry == 'reflectance':
        bt, unit = n.BT, 'Kelvin'
        for i in range(len(bands_tir)):
            band = level1[f'radiance_{i+1}']
            invalid = band.isnull()
            level1[f'radiance_{i+1}'] = calibrate_bt(band, i, granule_mtd, invalid, LUT_file)
    else: 
        bt, unit = n.Ltoa_tir, raw_data['radiance_1'].units
    rename = {f'radiance_{i+1}':bt+f'_{i+1}' for i in range(len(bands_tir))}
    level1 = level1.rename_vars(rename)
        
            
    if not split:
        level1 = merge(level1, dim='bands', pattern=r'(.+)_(\d+)')
        level1[bt].attrs = {}
        level1[bt].attrs['units'] = unit
        
    level1 = level1.drop_indexes(list(level1.coords)) \
                   .reset_coords(drop=True)
    return level1

def supplement_latlon(l1, chunks): 
        
    # Compute LatLon variables
    size = l1.cloud.shape
    latlon = [s.strip().split(' ') for s in l1.Boundary[10:-2].split(',')]
    latlon = np.array(latlon).astype(float)
    border = np.array((np.min(latlon,axis=0), np.max(latlon,axis=0)))
    step = (border[1]-border[0])/size

    lat = da.arange(border[0,1],border[1,1],step[1])
    lon = da.arange(border[0,0],border[1,0],step[0])
    lat = lat[:size[0]].reshape((size[0],1))
    lon = lon[:size[1]].reshape((1,size[1]))
    l1[n.lat] = xr.DataArray(da.repeat(lat, size[1], axis=1), 
                             dims = [n.rows,n.columns]).chunk(chunks=chunks)
    l1[n.lon] = xr.DataArray(da.repeat(lon, size[0], axis=0), 
                             dims = [n.rows,n.columns]).chunk(chunks=chunks)
    
    return l1

def calibrate_bt(array, band_index, granule_mtd, flags, LUT_file):
    """Calibration for the emissive channels."""
    
    if LUT_file:
                
        def find(Ltoa):
            if 0 <= Ltoa and Ltoa <= 60:
                indexLUT = Ltoa // 0.001
                radiance_x0 = indexLUT * 0.001
                radiance_x1 = radiance_x0 + 0.001 
                factor0 = (radiance_x1 - Ltoa) / 0.001
                factor1 = (Ltoa - radiance_x0) / 0.001     
                return (factor0 * LUT[indexLUT]) + (factor1 * LUT[indexLUT + 1])
            else: return 0

        # Interpolate the LUT values for each radiance based on two nearest LUT values        
        LUT = LUT_file[f'radiance_{band_index+1}']
        bt = np.vectorize(find)(array)     
        return bt

    else:
        # Initialized constants
        K1 = 1.191042 * 1e8
        K2 = 1.4387752 * 1e4
        
        # Temperature correction
        cwvl   = bands_tir[band_index] * 1e-3 
        gain   = granule_mtd.CalibrationGainCorrection[band_index]
        offset = granule_mtd.CalibrationOffsetCorrection[band_index]

        # Some versions of the modis files do not contain all the bands.
        array = xr.where(flags, K2 / (cwvl * np.log(K1 / (array * cwvl ** 5) + 1)), array)
        return xr.where(flags, gain * array + offset, array) 
    
def parse_attrs(stack, out_dic={}):
    current = [elem.strip() for elem in stack[0]]
    if len(stack) == 1: return out_dic
    if 'END' in current[0]:
        return out_dic, stack
    elif current[0] in ['GROUP','OBJECT']:
        out_dic[current[1]], new_stack = parse_attrs(stack[1:],{})
        return parse_attrs(new_stack[1:], out_dic)
    else:
        out_dic[current[0]] = current[1]
        return parse_attrs(stack[1:], out_dic)
    

def get_sample():
    product_name = 'ECOv002_L1CG_RAD_example.h5'
    return download_nextcloud(product_name, load_config()['dir_samples'], 'SampleData')