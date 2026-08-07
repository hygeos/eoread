from eoread.flags import GenericFlags, FlagsReaderBase
from eoread.tools import (
    filter_metadata, 
    spatial_resample, 
    collect_sample, 
    format_chunks
)

from typing import Union, Literal
from pathlib import Path

from core import log
from core.geo.naming import names
from core.tools import drop_unused_dims, merge, only

import xarray as xr
import dask.array as da


user_guide = 'https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/M1054E_PUG_2022_1005_V6.2.2_Terra_V6.2.3_Aqua.pdf'

bnames = [1,2,3,4,5,6,7,8,9,10,11,12,13,13.5,14,14.5,15,16,17,18,19,20,21,22,
          23,24,25,26,27,28,29,30,31,32,33,34,35,36]

cwvl = [650, 860, 470, 555, 1240, 1640, 2130, 410, 440, 485, 530, 550, 670, 672, 
        680, 682, 750, 870, 900, 935, 940, 3750, 3960, 3962, 4050, 4460, 4510, 
        1375, 6710, 7230, 8550, 9730, 11000, 12000, 13230, 13630, 13930, 14230]

# Planck's law constants 
h = 6.6260755e-34
c = 2.9979246e+8 # (meters per second)
k = 1.380658e-23 # (Joules per Kelvin)

# derived constants
K1 = 2.0 * h * c * c
K2 = h * c / k


def Level1_MODIS(
        filepath: Union[Path, str], 
        chunks: int = 100,
        resolution: Literal[250,500,1000,None] = 250,
        metadata_template: Union[list, None] = None,
        v1_compat: bool = False,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a MODIS Level1B product as an xarray.Dataset.
    
    MODIS (Moderate Resolution Imaging Spectroradiometer) provides 36 spectral
    bands from visible to thermal infrared with spatial resolutions of 250m,
    500m, and 1000m.

    Parameters
    ----------
    filepath : Path or str
        Path to the MODIS HDF4 file (.hdf).
    chunks : int, optional
        Size of chunks for spatial dimensions. Default is 100.
    resolution : {250, 500, 1000, None}, optional
        Resample all bands to provided resolution and concatenate.
        If None, keep original resolutions (250m, 500m, 1km) separate. Default is 250.
    metadata_template : list or None, optional
        List of metadata keys to include. If None, includes all metadata.
        Use empty list [] for minimal metadata. Default is None.
    v1_compat : bool, optional
        If True, formats output to match version 1 structure. Default is False.
    verbose : bool, optional
        If True, print debug information. Default is True.
        
    Examples
    --------
    >>> ds = Level1_MODIS('MOD021KM.A2023001.0000.061.*.hdf')
    """
    
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    
    # Format chunks
    chunks = format_chunks(chunks)
    
    # Revize variables
    if verbose: log.debug('Reading h4file')
    try:
        l1 = xr.open_dataset(filepath, engine='netcdf4')
    except OSError as e:
        if '' in str(e):
            raise ImportError('This error occurs because the netcdf4 library in'
            ' your current environment was compiled without HDF4 support. '
            'Install netcdf4 package with conda-forge.')
        else: 
            raise e
        
    l1 = l1.rename_vars({
        'Latitude': str(names.lat), 'Longitude': str(names.lon),
        'SensorZenith': str(names.vza)+'_tie', 'SensorAzimuth': str(names.vaa)+'_tie',
        'SolarZenith': str(names.sza)+'_tie', 'SolarAzimuth': str(names.saa)+'_tie'
    })
 
    # Read metadata
    metadata = {}
    if verbose: log.debug('parsing metadata text')
    for name in ['CoreMetadata.0','ArchiveMetadata.0','StructMetadata.0']:
        p = _Internal.parser_attrs(l1.attrs[name].split('\n'))
        p.parse()
        metadata.update(p.data)
    
    # Add band information
    if verbose: log.debug('Add central wavelength')
    l1 = l1.assign_coords({str(names.bands): da.array(bnames).astype(str)})
    l1 = l1.assign({str(names.cwav): ((str(names.bands)), cwvl)})

    # Change dimensions name and update coordinates
    l1 = _Internal.rename_dims(l1)
    
    # Change radiometry of input data   
    if verbose: log.debug('Read top of atmosphere data')
    l1 = _Internal.transform_radiometry(l1, chunks, resolution)
    l1 = _Internal.aggregate_vars(l1, chunks, resolution)
    l1 = _Internal.compute_bt(l1, resolution)
    
    # Upscale latlon variables
    if verbose: log.debug('upscale latlon variables')
    mapping = {k+'_red': k for k in [str(names.rows), str(names.columns)]}
    size = {k:v for k,v in l1.sizes.items() if k in mapping.values()}
    l1[str(names.lat)] = spatial_resample(l1[str(names.lat)].rename(mapping), size, chunks)
    l1[str(names.lon)] = spatial_resample(l1[str(names.lon)].rename(mapping), size, chunks)

    # Update attributes
    if verbose: log.debug('Add important attributes')
    attributes = l1.attrs
    l1.attrs = dict()
    l1.attrs[str(names.input_directory)] = str(filepath.parent)
    l1.attrs[str(names.resolution)]   = resolution
    l1.attrs[str(names.datetime)]     = metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['PRODUCTIONDATETIME']['VALUE'][1:-1]
    l1.attrs['night']             = str(metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['DAYNIGHTFLAG']['VALUE'] != '"Day"')
    l1.attrs[str(names.product_name)] = metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['LOCALGRANULEID']['VALUE'][1:-1]
    l1.attrs[str(names.platform)]     = metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDPLATFORMSHORTNAME']['VALUE'][1:-1]
    l1.attrs[str(names.sensor)]       = metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDSENSORSHORTNAME']['VALUE'][1:-1]
    l1.attrs[str(names.shortname)]    = metadata['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE'][1:-1]
    l1.attrs['version']           = int(metadata['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['VERSIONID']['VALUE'])
    l1.attrs['user_guide']        = user_guide
    l1.attrs['_flag_reader'] = 'eoread.modis.FlagsReader_MODIS'

    # SRF getter
    l1.attrs['_srf_getter'] = 'eotools.srf.get_SRF_eumetsat'
    l1.attrs['_srf_getter_arg'] = 'eos_1_modis'
    
    # Filter metadata if needed
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    metadata['attributes'] = attributes
    l1.attrs['metadata'] = filter_fn(metadata, metadata_template)
    
    # Reconstruct band groups
    l1 = drop_unused_dims(l1).chunk(chunks)
    band_dims = [b for b in l1.dims if b.startswith(str(names.bands)+'_')]
    bands = [list(l1[b].values) for b in band_dims]
    groups = [[b]*l1.sizes[b] for b in band_dims]
    bands, groups = sum(bands, []), sum(groups, [])
    groups = [groups[bands.index(str(b))] for b in l1[str(names.bands)].values]
    l1 = l1.assign_coords({str(names.bgroup): (str(names.bands), groups)})
    l1 = l1.transpose(..., str(names.rows), str(names.columns))

    return l1.unify_chunks()


def get_sample(level: int = 1) -> Path:
    return collect_sample(f'LEVEL{level}_MODISA_MYD', 'nasa', 'AQUA-MODIS-LR', level)
    

class FlagsReader_MODIS(FlagsReaderBase):
    """
    Flags reader for MODIS data.
    
    Provides access to MODIS quality flags for identifying invalid pixels.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return ['gflags']  # Use viewing zenith angle as reference
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return 'gflags'
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the MODIS dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            MODIS dataset containing gflags variable.
        flag_name : GenericFlags
            Standard flag identifier (currently only L1_INVALID supported).
        """
        if flag_name == GenericFlags.L1_INVALID:
            return ds['gflags']
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")


################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def transform_radiometry(level1: xr.Dataset, chunks: list, resolution: int) -> xr.Dataset:
        """Convert raw DN values to calibrated radiance and reflectance."""
        dim = 'dim'
        
        data_arrays = {'ltoa': [], 'rtoa': []}
        variables = ['EV_250_Aggr1km_RefSB','EV_500_Aggr1km_RefSB','EV_1KM_RefSB','EV_1KM_Emissive']        
        for var in variables:
            
            # Compute radiance
            rad = level1[var]
            bands_dim = only([d for d in rad.dims if str(names.bands) in d])
            bands = level1[bands_dim].values.astype(str)

            # Broadcast scales and offsets with appropriate dimensions
            level1 = level1.assign_coords({bands_dim: bands})
            scale = xr.DataArray(rad.radiance_scales, dims=bands_dim)
            offset = xr.DataArray(rad.radiance_offsets, dims=bands_dim)

            ltoa = scale * rad + offset
            ltoa.attrs['desc'] = names.ltoa.desc
            ltoa.attrs['unit'] = names.ltoa.unit
            
            if resolution:
                ltoa = ltoa.rename({bands_dim: str(names.bands)})
                data_arrays['ltoa'].append(ltoa)
            else:
                level1 = level1.assign({var: ltoa})

            # Compute Reflectance        
            if any('Emissive' in d for d in list(rad.dims)):
                rtoa = xr.full_like(rad, da.nan)
            
            else:
                # Broadcast scales and offsets with appropriate dimensions
                scale = xr.DataArray(rad.reflectance_scales, dims=bands_dim)
                offset = xr.DataArray(rad.reflectance_offsets, dims=bands_dim)

                rtoa = scale * rad + offset
                rtoa.attrs['desc'] = names.rtoa.desc
                rtoa.attrs['unit'] = names.rtoa.unit
                
            band_dim = str(names.bands) if resolution else bands_dim
            rtoa = rtoa.rename({bands_dim: band_dim})
            
            if resolution:
                data_arrays['rtoa'].append(rtoa)
            else:
                level1 = level1.assign({var: rtoa})

        # Combine into one dimension
        if resolution:
            level1[str(names.ltoa)] = xr.concat(data_arrays['ltoa'], dim=str(names.bands))
            level1[str(names.ltoa)] = level1[str(names.ltoa)].chunk(chunks)
            level1[str(names.rtoa)] = xr.concat(data_arrays['rtoa'], dim=str(names.bands))
            level1[str(names.rtoa)] = level1[str(names.rtoa)].chunk(chunks)
            level1 = level1.drop_vars(variables)
        
        return level1

    @staticmethod
    def aggregate_vars(l1: xr.Dataset, chunks: list, resolution: int) -> xr.Dataset:
        """Combine uncertainty indices into a single variable."""
        # Combine uncertainty in a single variable
        uncert_label = 'Uncert_Indexes'
        uncert_names = [d for d in l1.variables if uncert_label in d][:4]
        uncert_vars = [l1[v].rename({l1[v].dims[0] : str(names.bands)}) for v in uncert_names]
        
        if resolution:
            uncert = xr.concat(uncert_vars, dim=str(names.bands))
            l1[uncert_label] = uncert.drop_vars(str(names.bands)).chunk(chunks)
            l1 = l1.drop_vars(uncert_names)
        else:
            for v in uncert_vars:
                l1 = l1.assign()
        
        return l1

    @staticmethod
    def rename_dims(l1: xr.Dataset) -> xr.Dataset:
        """Rename HDF4 dimension names to standard naming convention."""
        revize_dims = {
            '2*nscans:MODIS_SWATH_Type_L1B': str(names.rows)+'_red', 
            '1KM_geo_dim:MODIS_SWATH_Type_L1B': str(names.columns)+'_red', 
            '10*nscans:MODIS_SWATH_Type_L1B': str(names.rows), 
            'Max_EV_frames:MODIS_SWATH_Type_L1B': str(names.columns),
            'Band_1KM_Emissive:MODIS_SWATH_Type_L1B': str(names.bands) + '_1KM_Emissive',
            'Band_250M:MODIS_SWATH_Type_L1B': str(names.bands) + '_250M',
            'Band_500M:MODIS_SWATH_Type_L1B': str(names.bands) + '_500M', 
            'Band_1KM_RefSB:MODIS_SWATH_Type_L1B': str(names.bands) + '_1KM_RefSB',
            'Band_250M': str(names.bands) + '_250M',
            'Band_500M': str(names.bands) + '_500M',
            'Band_1KM_RefSB': str(names.bands) + '_1KM_RefSB',
            'Band_1KM_Emissive': str(names.bands) + '_1KM_Emissive',
        }
            
        l1 = l1.rename(revize_dims)
        
        return l1

    @staticmethod
    def compute_bt(ds: xr.Dataset, resolution: int) -> xr.Dataset:
        """Compute brightness temperature from radiance using Planck's law with temperature corrections."""
        # Planck's law constants 
        h = 6.6260755e-34
        c = 2.9979246e+8 # (meters per second)
        k = 1.380658e-23 # (Joules per Kelvin)

        # derived constants
        K1 = 2.0 * h * c * c
        K2 = h * c / k

        # Effective central wavenumber (inverse centimeters)
        cwn = xr.DataArray(da.array([
            2.641775E+3, 2.505277E+3, 2.518028E+3, 2.465428E+3,
            2.235815E+3, 2.200346E+3, 1.477967E+3, 1.362737E+3,
            1.173190E+3, 1.027715E+3, 9.080884E+2, 8.315399E+2,
            7.483394E+2, 7.308963E+2, 7.188681E+2, 7.045367E+2],
            dtype=float), dims=(str(names.bands)))

        # Temperature correction slope (no units)
        tcs = xr.DataArray(da.array([
            9.993411E-1, 9.998646E-1, 9.998584E-1, 9.998682E-1,
            9.998819E-1, 9.998845E-1, 9.994877E-1, 9.994918E-1,
            9.995495E-1, 9.997398E-1, 9.995608E-1, 9.997256E-1,
            9.999160E-1, 9.999167E-1, 9.999191E-1, 9.999281E-1],
            dtype=float), dims=(str(names.bands)))

        # Temperature correction intercept (Kelvin)
        tci = xr.DataArray(da.array([
            4.770532E-1, 9.262664E-2, 9.757996E-2, 8.929242E-2,
            7.310901E-2, 7.060415E-2, 2.204921E-1, 2.046087E-1,
            1.599191E-1, 8.253401E-2, 1.302699E-1, 7.181833E-2,
            1.972608E-2, 1.913568E-2, 1.817817E-2, 1.583042E-2],
            dtype=float), dims=(str(names.bands)))
        
        # Transfer wavenumber [cm^(-1)] to wavelength [m]
        cwvl = 1. / (cwn * 100)
        bands_emis = ds[str(names.bands)+'_1KM_Emissive'].values.astype(str)
        offset = xr.Dataset({'cwvl': cwvl, 'tcs': tcs, 'tci': tci})
        offset = offset.assign_coords({str(names.bands): bands_emis})
        
        if resolution:
            bands, var = str(names.bands), str(names.ltoa)
        else :
            bands, var = str(names.bands)+'_1KM_Emissive', 'EV_1KM_Emissive'
        
        for b in ds[bands].values:
            array = ds[var].sel({bands: b})
            if b not in offset[str(names.bands)]:
                ds[str(names.bt)+f'_{b}'] = xr.full_like(array, da.nan)
                continue
            off = offset.sel({str(names.bands): b})
            array = K2 / (off['cwvl'] * da.log(K1 / (1e6 * array * off['cwvl'] ** 5) + 1))
            ds[str(names.bt)+f'_{b}'] = ((array - off['tci']) / off['tcs'])
            
        ds = merge(ds, bands, pattern=f'({str(names.bt)})'+r'_(.+)', dtype=str)
        ds[str(names.bt)].attrs['unit'] = 'Kelvin'
        return ds

    class parser_attrs:
        """Parser for MODIS HDF4 metadata attributes."""
        
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
                    
            self.text = self.text[1:]
            
            line = line.strip()
            while self.is_void(line):
                line = self.consume()
            
            return line.strip()
        
        def peek(self) -> str:
            """Return the next non-empty line without consuming it."""
            line = self.text[0].strip()
            while self.is_void(line): 
                self.text = self.text[1:]
                line = self.text[0].strip()
            return line
        
        def is_void(self, line: str) -> bool:
            """Check if a line is empty."""
            return len(line) == 0
        
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
            
            key, val = [i.strip() for i in line.split('=')]
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