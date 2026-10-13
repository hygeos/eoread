from eoread.tools import filter_metadata, format_chunks, collect_sample
from eoread.flags import GenericFlags, FlagsReaderBase

from core.geo.naming import names
from core.tools import merge
from core import env, log

from dask.array import meshgrid
from typing import Union
from pathlib import Path
from re import findall

import numpy as np
import xarray as xr


user_guide = 'https://ecostress.jpl.nasa.gov/downloads/userguides/1_ECOSTRESS_L1_UserGuide_20190619.pdf'

def Level1C_ECOSTRESS(
        filepath: Union[Path, str], 
        chunks: Union[int, list, dict] = 500, 
        metadata_template: Union[list, None] = None,
        v1_compat: bool = False,
        verbose: bool = True,
    ) -> xr.Dataset:
    """
    Read an ECOSTRESS Level1 product as an xarray.Dataset.
    
    ECOSTRESS (Ecosystem Spaceborne Thermal Radiometer Experiment on Space Station)
    provides high-resolution thermal infrared measurements across 5 spectral bands.

    Parameters
    ----------
    filepath : Path or str
        Path to the ECOSTRESS HDF5 file (.h5).
    chunks : int, list, or dict, optional
        Size of chunks for spatial dimensions. If int, applies to both dimensions.
        If list, should be [rows_chunk, columns_chunk]. Default is 500.
    metadata_template : list or None, optional
        List of metadata keys to include. If None, includes all metadata.
        Use empty list [] for minimal metadata. Default is None.
    v1_compat : bool, optional
        If True, formats output to match version 1 structure. Default is False.
    verbose : bool, optional
        If True, print debug information. Default is True.
        
    Examples
    --------
    >>> ds = Level1_ECOSTRESS('ECOv002_L1CG_RAD_*.h5', chunks=1000)
    """
    
    # Check that file exists
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    
    # Format chunks
    chunks = format_chunks(chunks)
    
    # Read provided file
    if verbose: log.debug('Reading h5file')
    data = xr.open_datatree(filepath, phony_dims='sort', engine='h5netcdf')
    raw = data['HDFEOS/GRIDS/ECO_L1CG_RAD_70m/Data Fields']
    raw = raw.to_dataset().rename(
        phony_dim_2=str(names.rows), phony_dim_3=str(names.columns)
    ).chunk(chunks=chunks)
    
    # Extract and parse metadata
    granule_mtd = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata']
    attributes = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata']
    
    if verbose: log.debug('parsing metadata text')
    info = data['HDFEOS INFORMATION']['StructMetadata.0'].values.item().decode()
    p = _Internal.metadata_parser(info.split('\n'))
    p.parse()
    
    # Compute Brightness Temperature and Radiance
    if verbose: log.debug('compute brightness temperature')
    l1 = _Internal.transform_radiometry(raw, granule_mtd)   
    
    # Add attributes
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    if verbose: log.debug('add important attributes')
    l1.attrs['_flag_reader'] = 'eoread.ecostress.FlagsReader_ECOSTRESS'
    l1.attrs['metadata'] = {k: v.item() for k,v in attributes.items()}
    l1.attrs['metadata'] = filter_fn(l1.attrs['metadata'], metadata_template)
    l1.attrs['hdfeos_info'] = filter_fn(p.data, metadata_template)
    l1.attrs['user_guide'] = user_guide
    l1.attrs['daytime'] = bool(attributes['DayNightFlag'] == 'Day')
    
    # Add general information
    l1.attrs[str(names.datetime)] = l1.metadata['ProductionDateTime']
    l1.attrs[str(names.platform)] = l1.metadata['PlatformShortName']
    l1.attrs[str(names.sensor)] = l1.metadata['InstrumentShortName']
    l1.attrs[str(names.product_name)] = filepath.name
    l1.attrs[str(names.input_directory)] = str(filepath.parent)
    l1.attrs[str(names.resolution)] = 70
    
    # SRF getter
    l1.attrs['_srf_getter'] = 'eotools.srf.get_SRF_eumetsat'
    l1.attrs['_srf_getter_arg'] = 'iss_1_ecostres'
     
    # Update coordinates
    l1 = l1.assign_coords({
        str(names.bands): l1[str(names.bands)].values.astype(str),
        str(names.bgroup): (str(names.bands), ['bands_ir']*5)
    })
    
    # Add latlon variables
    if verbose: log.debug('add latlon variables')
    l1 = _Internal.supplement_latlon(l1, chunks)
    
    if v1_compat: return _v1_compat(l1)
    else: return l1


def Level2_ECOSTRESS(
        filepath: Union[Path, str], 
        chunks: Union[int, list, dict] = 500, 
        metadata_template: Union[list, None] = None,
        verbose: bool = True,
    ) -> xr.Dataset:
    """
    Read an ECOSTRESS Level2 product as an xarray.Dataset.
    
    Processes Level2 Land Surface Temperature and Emissivity (LSTE) data.

    Parameters
    ----------
    filepath : Path or str
        Path to the ECOSTRESS Level2 HDF5 file (.h5).
    chunks : int, list, or dict, optional
        Size of chunks for spatial dimensions. Default is 500.
    metadata_template : list or None, optional
        List of metadata keys to include. If None, includes all metadata. Default is None.
    verbose : bool, optional
        If True, print debug information. Default is True.

    Returns
    -------
    xr.Dataset
        xarray.Dataset containing:
        - Surface temperature and emissivity products
        - Quality masks and flags
        - Geolocation arrays (lat, lon)
        - Metadata attributes
    """
    # Revise variables
    filepath = Path(filepath)
    data = xr.open_datatree(filepath, phony_dims='sort', engine='h5netcdf')
    raw = data['HDFEOS/GRIDS/ECO_L2G_LSTE_70m/Data Fields']
    l2 = raw.to_dataset().chunk(chunks=chunks)
    
    # Read Metadata
    granule_mtd = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata']
    attributes  = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata']
    
    info = data['HDFEOS INFORMATION']['StructMetadata.0'].values.item().decode()
    p = _Internal.metadata_parser(info.split('\n'))
    p.parse()
    
    # Change radiometry of input data 
    l2 = _Internal.transform_radiometry(raw, granule_mtd)
    
    # Add attributes
    for att in list(attributes):
        l2.attrs[att] = attributes[att].values.item()
    l2.attrs['hdfeos_info'] = p.data
    l2.attrs['user guide'] = user_guide
    
    # Change dimensions name and update coordinates
    new_dims = [str(names.rows), str(names.columns)]    
    revize_dims = dict(zip(list(l2.dims), new_dims))
    l2 = l2.rename_dims(revize_dims)
    
    # Add latlon variables
    l2 = _Internal.supplement_latlon(l2, chunks)
    return l2


def get_sample(level: int = 1) -> Path:
    """
    Download or retrieve a sample ECOSTRESS product for testing.
    
    Requires the 'sand' module for NASA data access.

    Parameters
    ----------
    level : int, optional
        Processing level of the product (1 or 2). Level 1 provides
        radiance/brightness temperature, Level 2 provides surface temperature.
        Default is 1.

    Returns
    -------
    Path
        Path to the downloaded ECOSTRESS HDF5 file.
        
    Raises
    ------
    ImportError
        If the 'sand' module is not installed.
    """
    return collect_sample(f'LEVEL{level}_ECOSTRESS', 'nasa', 'ISS-ECOSTRESS', level)


class FlagsReader_ECOSTRESS(FlagsReaderBase):
    """
    Flags reader for ECOSTRESS data products.
    
    Provides standardized access to ECOSTRESS quality flags including land/water mask,
    data quality, cloud mask, and invalid data detection.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return ['water', str(names.ltoa)]  # Use viewing zenith angle as reference
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return 'water'
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the ECOSTRESS dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            ECOSTRESS dataset containing flag variables.
        flag_name : GenericFlags
            Standard flag identifier (L1_INVALID, LAND, L1_DEGRADED, or CLOUD).
        """
        if flag_name == GenericFlags.L1_INVALID:
            # L1_INVALID is True where vza is NaN (invalid data)
            band = ds[str(names.ltoa)].sel({str(names.bands): '4'}).isnull()
            return xr.DataArray(band, dims=band.dims, coords=band.coords)
        elif flag_name == GenericFlags.LAND:
            return xr.DataArray(
                ~ds['water'],
                dims=ds['water'].dims,
                coords=ds['water'].coords
            )
        elif flag_name == GenericFlags.L1_DEGRADED:
            mask = ds['data_quality'].max(str(names.bands))
            return xr.DataArray(mask, dims=mask.dims, coords=mask.coords)
        elif flag_name == GenericFlags.CLOUD:
            return xr.DataArray(
                ds['cloud'],
                dims=ds['cloud'].dims,
                coords=ds['cloud'].coords
            )
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")


################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def transform_radiometry(raw_data: xr.Dataset, granule_mtd: xr.Dataset) -> xr.Dataset:
        """Convert raw radiances to calibrated units and compute brightness temperature."""
        # Combine band radiances into a single variable 
        level1 = merge(raw_data, dim=str(names.bands), pattern=r'(.+)_(\d+)')
        
        # Rename radiance variable
        level1 = level1.rename({'radiance': str(names.ltoa)})
        level1[str(names.ltoa)].attrs['unit'] = 'W/sr/m^2'
        
        # Compute brightness temperature for Emissive bands 
        level1 = _Internal.compute_bt(level1, granule_mtd)
        return level1

    @staticmethod
    def parse_wkt(wkt: str) -> list:
        """Parse Well-Known Text (WKT) geometry string to extract coordinate pairs."""
        points = findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', wkt)
        points = [(float(lon), float(lat)) for lon, lat in points]
        return np.array(points)
    
    @staticmethod
    def supplement_latlon(l1: xr.Dataset, chunks: dict) -> xr.Dataset:
        """Add latitude and longitude coordinates based on scene boundary."""
        # Compute LatLon variables
        size = l1['cloud'].sizes
        coords = _Internal.parse_wkt(l1.metadata['SceneBoundaryLatLonWKT'])
        north  = coords[:,1].max()
        south  = coords[:,1].min()
        east   = coords[:,0].max()
        west   = coords[:,0].min()
        
        # Build the lat and lon arrays
        lat = np.linspace(north, south, size[str(names.rows)])
        lon = np.linspace(west, east, size[str(names.columns)])
        lon, lat = meshgrid(lon, lat)
        
        dims = list(size)
        l1[str(names.lon)] = xr.DataArray(lon, dims=dims).chunk(chunks=chunks)
        l1[str(names.lat)] = xr.DataArray(lat, dims=dims).chunk(chunks=chunks)
        return l1

    @staticmethod
    def compute_bt(l1: xr.Dataset, granule_mtd: xr.Dataset) -> xr.Dataset:
        """Compute brightness temperature from radiance using Planck's law."""
        # Initialized constants
        K1 = 1.191042 * 1e8
        K2 = 1.4387752 * 1e4
        
        # Temperature correction
        cwvl = granule_mtd.BandSpecification[1:].rename(phony_dim_0=str(names.bands))
        gain = granule_mtd.CalibrationGainCorrection.rename(phony_dim_1=str(names.bands))
        offset = granule_mtd.CalibrationOffsetCorrection.rename(phony_dim_1=str(names.bands))
        l1 = l1.assign({str(names.cwav): ((str(names.bands)), cwvl.data*1e3)}) # convert into nm
        
        # Some versions of the modis files do not contain all the bands.
        valid = ~l1[str(names.ltoa)].isnull()
        array = gain * l1[str(names.ltoa)].where(valid) + offset
        l1[str(names.bt)] = K2 / (cwvl * np.log(K1 / (array.where(valid) * cwvl ** 5) + 1))
        l1[str(names.bt)].attrs = {'unit': 'Kelvin'}
        
        return l1

    class metadata_parser:
        """Parser for HDFEOS metadata structure."""
        
        def __init__(self, text: list):
            """Initialize parser with metadata text lines."""
            self.data = {}
            self.text = text.copy()
        
        def empty(self) -> bool:
            """Check if there are remaining lines to parse."""
            return len(self.text) == 0
        
        def consume(self) -> str:
            """Remove and return the next non-empty line."""
            
            line = self.text[0]
            if len(self.text) == 1: # case for last line
                self.text = []
                return line.strip()
                    
            self.text = self.text[1:]
            
            line = line.strip()
            while line == "":
                line = self.consume()
            
            return line.strip()
        
        def peek(self) -> str:
            """Return the next line without consuming it."""
            return self.text[0].strip()
        
        def parse(self) -> None:
            """Parse all remaining lines into structured metadata."""
            
            while not self.empty():
                end = self._parse_recu(self.data)
                if end: break
                
        def _parse_recu(self, data: dict = None) -> bool:
            """Recursively parse metadata groups and objects."""
            line = self.consume()
            if line == "END":
                return True
            
            key, val = line.split('=')
            if key in ['GROUP','OBJECT']: 
                data[val] = {}
                
                line = self.peek()
                while f'END_' not in line:
                    self._parse_recu(data[val])
                    line = self.peek() # refresh peeked line !! 
                
                # closing tag
                self.consume()
            
            else: data[key] = val 
            
            return False