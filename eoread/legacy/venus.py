#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://www.eoportal.org/satellite-missions/venus#vssc-ven%C2%B5s-superspectral-camera

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from pyproj import Proj

from core.files import mdir
from core.table import read_xml
from core.geo.naming import names
from core.tools import merge, drop_unused_dims
from core import env, log

from eoread.flags import GenericFlags, FlagsReaderBase
from eoread.tools import (
    open_raster, 
    spatial_resample, 
    filter_metadata, 
    format_chunks
)


user_guide = 'https://www.cesbio.cnrs.fr/multitemp/ven%c2%b5s-product-format/'

def Level1_VENUS(
        dirname: Union[str, Path], 
        chunks: Union[int, tuple] = 500,
        metadata_template: Union[list, None] = None,
        read_masks: bool = True,
        v1_compat: bool = False, 
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a VENµS Level1C product as an xarray.Dataset.
    
    Formats the Dataset to contain TOA reflectances, viewing/solar angles
    on the full grid, and geolocation information.
    
    VENµS (Vegetation and Environment monitoring on a New Micro-Satellite) provides
    12 superspectral bands from 420nm to 910nm with 5m spatial resolution.

    Parameters
    ----------
    dirname : str or Path
        Path to the VENµS product directory.
    chunks : int or tuple, optional
        Size of chunks for spatial dimensions. If int, applies to both dimensions.
        If tuple, should be (rows_chunk, columns_chunk). Default is 500.
    metadata_template : list or None, optional
        List of metadata keys to include. If None, includes all metadata. Default is None.
    read_masks : bool, optional
        If True, reads compressed quality masks (PIX, SAT, CLD, USI).
        Warning: Uncompressing masks is time-consuming. Default is True.
    v1_compat : bool, optional
        If True, formats output to match version 1 structure. Default is False.
    verbose : bool, optional
        If True, print debug information. Default is True.
    
    Raises
    ------
    AssertionError
        If the directory does not exist.
        
    Examples
    --------
    >>> ds = Level1_VENUS('VENUS-XS_*_L1C_*', chunks=1000)
    >>> print(ds.Rtoa.sel(bands='B8'))  # Red edge band
    """
    
    # Check that folder exists
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Folder does not exists'
    
    # Format chunks
    chunks = format_chunks(chunks)
    
    # read metadata
    if verbose: log.debug('Reading metadata')
    ds, metadata_granule = _Internal.read_metadata(ds, dirname, metadata_template)

    # read geaometry
    if verbose: log.debug('Read and compute geometric angles')
    ds = _Internal.read_geometry(ds, dirname, chunks)

    # read TOA
    if verbose: log.debug('Read top of atmosphere data')
    radio_info = metadata_granule['Radiometric_Informations']
    quantif = float(radio_info['REFLECTANCE_QUANTIFICATION_VALUE'])
    ds = _Internal.read_toa(ds, dirname, quantif, chunks)

    # lat-lon
    if verbose: log.debug('Compute LatLon raster')
    geocoding = metadata_granule['Geoposition_Informations']
    ds = _Internal.supplement_latlon(ds, chunks, geocoding)
    
    # read cloud altitude
    if verbose: log.debug('Open masks')
    ratio = {str(names.columns): ds.totalwidth, str(names.rows): ds.totalheight} 
    cld = open_raster(dirname/'DATA', '*CLA_ALL.tif', engine='rasterio')
    cld = cld.rename(x=str(names.columns), y=str(names.rows))
    ds['CLA_ALL'] = spatial_resample(cld, ratio, chunks, 'repeat').T
    
    if read_masks:
        
        # read cloud mask
        kwargs = dict(compress_ext='.zip', engine='rasterio')
        cld = open_raster(dirname/'MASKS','*CLD_XS.zip', **kwargs).chunk(chunks)
        ds['CLD_XS'] = cld.rename(x=str(names.columns), y=str(names.rows))
        
        # read cloud mask
        usi = open_raster(dirname/'MASKS','*USI_XS.zip', **kwargs).chunk(chunks)
        ds['USI_XS'] = usi.rename(x=str(names.columns), y=str(names.rows))
    
        # Read quality masks
        for bn in ds[str(names.bands)]:
            
            pix = open_raster(dirname/'MASKS',f'*PIX_{bn.values}.zip', **kwargs).chunk(chunks)
            ds[f'PIX_{bn.values}'] = pix.rename(x=str(names.columns), y=str(names.rows))
            
            sat = open_raster(dirname/'MASKS',f'*SAT_{bn.values}.zip', **kwargs).chunk(chunks) 
            ds[f'SAT_{bn.values}'] = sat.rename(x=str(names.columns), y=str(names.rows))
    
        # Concatenate quality masks 
        ds = merge(ds, str(names.bands), pattern=r'(.+)_(B.+)', dtype=str)    

    elif verbose:
        log.debug('Masks are not red due to uncompression time consuming. '
                  'Active option read_masks to read them')    
    
    # Reconstruct band groups
    ds = drop_unused_dims(ds)
    groups = ['bands_vnir']*len(ds[str(names.bands)])
    ds = ds.assign_coords({str(names.bgroup): (str(names.bands), groups)})
    
    # Update spatial coordinates
    ds = ds.assign_coords({
        str(names.columns): np.arange(len(ds[str(names.columns)])),
        str(names.rows): np.arange(len(ds[str(names.rows)]))
    })
    
    # Add flag reader and SRF getter in attributes
    ds.attrs['_flag_reader'] = 'eoread.venus.FlagsReader_VENUS'
    ds.attrs['_srf_getter'] = 'eoread.venus.get_SRF'
    ds.attrs['_srf_getter_arg'] = None
    
    return ds.unify_chunks()


def Level2_VENUS(
        dirname: Union[str, Path], 
        chunks: Union[int, tuple] = 500,
        metadata_template: Union[list, None] = None
    ) -> xr.Dataset:
    """
    Read a VENµS Level2A product as an xarray.Dataset.
    
    Processes Level2A surface reflectance products with atmospheric correction.

    Parameters
    ----------
    dirname : str or Path
        Path to the VENµS Level2A product directory.
    chunks : int or tuple, optional
        Size of chunks for spatial dimensions. If int, applies to both dimensions.
        Default is 500.
    metadata_template : list, optional
        List of metadata keys to include. If None, includes all metadata.
        Default is None.

    Returns
    -------
    xr.Dataset
        Dataset containing surface reflectance, geometry, and quality flags.
    """
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Folder does not exists'
    
    # Format chunks
    chunks = format_chunks(chunks)
    
    # read metadata
    log.debug('Reading metadata')
    ds, metadata_granule = _Internal.read_metadata(ds, dirname, metadata_template)
    
    # lat-lon
    log.debug('Compute LatLon raster')
    geocoding = metadata_granule['Geoposition_Informations']
    ds = _Internal.supplement_latlon(ds, chunks, geocoding)

    # read geaometry
    log.debug('Read and compute geometric angles')
    ds = _Internal.read_geometry(ds, dirname, chunks)

    # read reflectances
    radio_info = metadata_granule['Radiometric_Informations']
    quantif = float(radio_info['REFLECTANCE_QUANTIFICATION_VALUE'])
    ds = _Internal.read_rho(ds, dirname, quantif, chunks)
    
    # read cloud mask
    log.debug('Open masks')
    cld = open_raster(dirname/'MASKS','*CLM_XS.tif', engine='rasterio').chunk(chunks)
    ds['CLM_XS'] = cld.rename(x=str(names.columns), y=str(names.rows))
    
    # read other masks
    usi = open_raster(dirname/'MASKS','*USI_XS.tif', engine='rasterio').chunk(chunks)
    ds['USI_XS'] = usi.rename(x=str(names.columns), y=str(names.rows))
    
    cld = open_raster(dirname/'MASKS','*SAT_XS.tif', engine='rasterio').chunk(chunks)
    ds['SAT_XS'] = cld.rename(x=str(names.columns), y=str(names.rows))
    
    usi = open_raster(dirname/'MASKS','*PIX_XS.tif', engine='rasterio').chunk(chunks)
    ds['PIX_XS'] = usi.rename(x=str(names.columns), y=str(names.rows), band=str(names.bands))
    
    cld = open_raster(dirname/'MASKS','*IAB_XS.tif', engine='rasterio').chunk(chunks)
    ds['IAB_XS'] = cld.rename(x=str(names.columns), y=str(names.rows))
    
    usi = open_raster(dirname/'MASKS','*EDG_XS.tif', engine='rasterio').chunk(chunks)
    ds['EDG_XS'] = usi.rename(x=str(names.columns), y=str(names.rows))
    
    ds = drop_unused_dims(ds)
    groups = ['bands_vnir']*len(ds[str(names.bands)])
    ds = ds.assign_coords({str(names.bgroup): (str(names.bands), groups)})
    
    return ds.unify_chunks()


def get_sample(level: int = 1) -> Path:
    """
    Retrieve a sample VENµS product directory for testing.
    
    Returns paths to pre-configured sample products from environment variables.

    Parameters
    ----------
    level : int, optional
        Processing level (1 for Level1C, 2 for Level2A). Default is 1.

    Returns
    -------
    Path
        Path to the VENµS product directory.
        
    Raises
    ------
    ValueError
        If level is not 1 or 2.
    """
    
    # Check if user has provided a path
    variable = env.getvar(f'LEVEL{level}_VENUS', default='')
    
    # If not provided, try to download a sample with SAND
    if variable == '':
        
        # Check SAND importation
        try: 
            from sand.sample_product import products
            from sand.cnes import DownloadCNES
        except ImportError:
            raise ImportError('To use get_sample function, you need to install SAND module')
        
        # Retrieve name of example product
        sand_collection = 'VENUS'
        params = products[sand_collection]['constraint']
        
        # Download product with SAND
        dl = DownloadCNES()
        directory = env.getdir('DIR_SAMPLES')/sand_collection
        query = dl.query(collection_sand=sand_collection, level=level, **params)
        target = dl.download(query[0], directory)
        
        assert target.exists()
        return target
        
    else:
        return Path(variable)
    
    
def get_SRF(
    ds_in: Union[xr.Dataset, None] = None, 
    dir_data: Union[Path, None] = None
) -> xr.Dataset:
    """
    Load VENµS spectral response functions (SRF) for radiometric calculations.
    
    Downloads SRF data from the official repository if not already cached.

    Parameters
    ----------
    ds_in : xr.Dataset, optional
        Optional dataset with band names. If provided, output bands
        are referenced by ds_in.bands. Otherwise uses band IDs 1-12.
        Default is None.
    dir_data : Path, optional
        Directory to cache SRF data. If None, uses default static directory.
        Default is None.

    Returns
    -------
    xr.Dataset
        Dataset containing:
            - SRF curves for each VENµS band
            - wav: Wavelength coordinate in nanometers
            - Band variables named by band ID or from ds_in.bands

    Examples
    --------
        Load SRF and query at 550nm:

    >>> srf = get_SRF()
    >>> print(srf.sel(wav=550, method='nearest'))  # SRF at 550nm
    """
    from core.network.download import download_url
    from core.table import read_csv
    
    if dir_data is None:
        dir_data = mdir(env.getdir('DIR_STATIC')/'venus')

    url = 'https://labo.obs-mip.fr/wp-content-labo/uploads/sites/19/2018/09/rep6S.txt'
    srf_file = download_url(url, dir_data)
    nbands = 12
    ibands = range(1, nbands+1)
    df = read_csv(srf_file, sep=None, names=['wav_um', *ibands])

    ds = xr.Dataset()
    ds.attrs["desc"] = 'Spectral response functions for VENµS'

    if ds_in is None:
        bids = ibands
    else:
        assert len(ds_in.bands) == nbands
        bids = ds_in.bands.values
    for i in range(nbands):
        ds[bids[i]] = xr.DataArray(
            df[ibands[i]].values,
            dims=["wav"],
            attrs={"band_info": f"VENUS band {bids[i]}"},
        )

    ds = ds.assign_coords(wav=df['wav_um'].values*1000)
    ds[str(names.wav)].attrs["units"] = "nm"

    return ds


class FlagsReader_VENUS(FlagsReaderBase):
    """
    Flags reader for VENµS (Vegetation and Environment monitoring on a New Micro-Satellite) data.
    
    Provides access to cloud masks and data validity flags.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return ['CLA_ALL', 'CLD_XS']  # Use viewing zenith angle as reference
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return 'CLA_ALL'
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the VENµS dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            VENµS dataset containing CLA_ALL and CLD_XS variables.
        flag_name : GenericFlags
            Standard flag identifier (L1_INVALID or CLOUD).
        """
        if flag_name == GenericFlags.L1_INVALID:
            # L1_INVALID is True where vza is NaN (invalid data)
            return xr.DataArray(
                ds['CLA_ALL'].isnull(),
                dims=ds['CLA_ALL'].dims,
                coords=ds['CLA_ALL'].coords
            )
        elif flag_name == GenericFlags.CLOUD:
            return xr.DataArray(
                ds['CLD_XS'],
                dims=ds['CLD_XS'].dims,
                coords=ds['CLD_XS'].coords
            )
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")


################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def read_metadata(
            ds: xr.Dataset, 
            dirname: Path, 
            metadata_template: Union[list, None]
        ) -> tuple[xr.Dataset, dict]:
        """Extract metadata from XML files and populate dataset attributes."""
        
        # load xml file
        xmlfiles = list((dirname/'DATA').glob('*UII_ALL.xml'))
        assert len(xmlfiles) == 1
        xmlroot = read_xml(xmlfiles[0])

        # load main xml file
        xmlfiles = list(dirname.glob('*MTD_ALL.xml'))
        assert len(xmlfiles) == 1
        xmlgranule = read_xml(xmlfiles[0])
        
        # Extract resolution, band names and wavelength
        resolution = None
        bandnames, cwvl = [], []
        radio_info = xmlgranule['Radiometric_Informations']['Spectral_Band_Informations_List']
        for band in radio_info['Spectral_Band_Informations']:
            r = band['SPATIAL_RESOLUTION']['values']
            if resolution: assert resolution == r
            else: resolution = r
            cwvl.append(band['Wavelength']['CENTRAL']['values'])
            bandnames.append(band['attributes']['band_id'])
        ds = ds.assign({str(names.cwav): ((str(names.bands)), cwvl)})
        ds = ds.assign_coords({str(names.bands): bandnames})
        
        # read date
        date = xmlgranule['Product_Characteristics']['ACQUISITION_DATE']

        # get platform
        platform = xmlgranule['Product_Characteristics']['PLATFORM']
        assert platform == 'VENUS'

        # read image size for current resolution
        shape_info = xmlgranule['Geoposition_Informations']['Geopositioning']['Group_Geopositioning_List']
        ds.attrs['totalheight'] = shape_info['Group_Geopositioning']['NROWS']
        ds.attrs['totalwidth'] = shape_info['Group_Geopositioning']['NCOLS']
        
        # attributes
        filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
        ds.attrs[str(names.datetime)] = date
        ds.attrs[str(names.platform)] = platform
        ds.attrs[str(names.resolution)] = resolution
        ds.attrs[str(names.sensor)] = 'VENUS'
        ds.attrs[str(names.product_name)] = xmlgranule['Product_Characteristics']['PRODUCT_ID']
        ds.attrs[str(names.input_directory)] = str(dirname.parent)
        ds.attrs['metadata_granule'] = filter_fn(xmlgranule, metadata_template)
        ds.attrs['metadata'] = filter_fn(xmlroot, metadata_template)
        ds.attrs['user_guide'] = user_guide
        
        return ds, xmlgranule

    @staticmethod
    def supplement_latlon(ds: xr.Dataset, chunks: list, xmlgranule: dict) -> xr.Dataset:
        """Generate latitude and longitude arrays from corner coordinates."""
        
        # Get tile projection from metadata
        code = xmlgranule['Coordinate_Reference_System']['Horizontal_Coordinate_System']
        proj = Proj(code['HORIZONTAL_CS_CODE'])
        
        # lookup position in the UTM grid
        geopos = xmlgranule['Geopositioning']['Global_Geopositioning']
        latlon = np.array([[geo['LON'], geo['LAT']] for geo in geopos.values()])
        lat, lon = latlon[:,1], latlon[:,0]
                
        # Compute latitude and longitude rasters
        lon = np.linspace(lon.min(), lon.max(), ds.totalwidth)
        lat = np.linspace(lat.min(), lat.max(), ds.totalheight)
        lon, lat = np.meshgrid(lon, lat)
        
        # Add rasters in the dataset
        dims = [str(names.rows), str(names.columns)]
        return ds.assign({
            str(names.lon): xr.DataArray(lon, dims=dims).chunk(chunks), 
            str(names.lat): xr.DataArray(lat, dims=dims).chunk(chunks)
        })
            
    @staticmethod
    def read_toa(
            ds: xr.Dataset, 
            granule_dir: Path, 
            quantif: float, 
            chunks: list
        ) -> xr.Dataset:
        """Read and calibrate TOA reflectance from TIFF files."""
        for name in ds[str(names.bands)]:
            
            arr = open_raster(granule_dir, f'*REF_{name.values}.tif', engine='rasterio').chunk(chunks)
            arr = (arr/quantif).astype('float32')
            
            ratio = {str(names.rows): ds.totalheight, str(names.columns): ds.totalwidth}        
            arr_resampled = spatial_resample(arr, ratio, chunks)
            ds[str(names.rtoa)+f'_{name.values}'] = arr_resampled

        ds = merge(ds, dim=str(names.bands), pattern=r'(.+)_(B.+)', dtype=str)
        ds[str(names.rtoa)].attrs.update(unit=None)
        return ds

    @staticmethod
    def read_rho(
            ds: xr.Dataset, 
            granule_dir: Path, 
            quantif: float, 
            chunks: list
        ) -> xr.Dataset:
        """Read Level2 surface and flat reflectances, aerosol and water vapor."""
        for rho, var in zip(['SRE','FRE'],['rho_surface','rho_flat']):
            for name in ds[str(names.bands)]:
                
                arr = open_raster(granule_dir, f'*{rho}_{name.values}.tif', engine='rasterio').chunk(chunks)
                arr = (arr/quantif).astype('float32')

                ratio = {'y': ds.totalheight, 'x': ds.totalwidth}  
                arr_resampled = spatial_resample(arr, ratio, chunks)
                ds[var+f'_{name.values}'] = arr_resampled

        ds = merge(ds, dim=str(names.bands), pattern=r'(.+)_(B.+)', dtype=str)
        
        # read Aerosol_Optical_Thickness of waper vapor content
        atb = open_raster(granule_dir, '*ATB_XS.tif', engine='rasterio').chunk(chunks)
        ds['water_vapor'] = atb.sel(band=1)
        ds['aod'] = atb.sel(band=2)

        return ds
    
    @staticmethod
    def read_geometry(ds: xr.Dataset, dirname: Path, chunks: list) -> xr.Dataset:
        """Read solar and viewing angle grids from TIFF files."""
        # read solar angles
        sa = open_raster(dirname/'DATA','*SOL_ALL.tif', engine='rasterio').chunk(chunks)
        ds['SOL_ALL'] = sa.rename(x=str(names.columns)+'_tie', y=str(names.rows)+'_tie')
        
        # read view angles
        va = open_raster(dirname/'DATA','*VIE_ALL.tif', engine='rasterio').chunk(chunks)
        ds['VIE_ALL'] = va.rename(x=str(names.columns)+'_tie', y=str(names.rows)+'_tie')
        
        return ds.rename(band=str(names.bands)+'_angle')