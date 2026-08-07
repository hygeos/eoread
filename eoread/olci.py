#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np

from pathlib import Path
from typing import Literal, Union
from datetime import datetime
from warnings import filterwarnings
from re import findall

from core import log
from core.table import read_xml
from core.geo.naming import names
from core.tools import getflag, getflags

from eoread.flags import FlagsReaderBase, GenericFlags
from eoread.tools import (
    spatial_resample, 
    filter_metadata,
    collect_sample,
    format_chunks
)


# To filter warning message raised by instrument_data reading
filterwarnings('ignore', message=".*Duplicate dimension.*")


def Level1_OLCI(
        dirname: Union[str, Path], 
        chunks: Union[int, tuple] = 500,
        interp_angles: Literal['atan2', 'linear', 'legacy'] = 'linear',
        metadata_template: Union[list, None] = None, 
        v1_compat: bool = False,
        engine: str | None = None,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a Sentinel-3 OLCI Level1 product as an xarray.Dataset.
    
    OLCI (Ocean and Land Colour Instrument) provides 21 spectral bands from
    400nm to 1020nm with 300m spatial resolution.

    Parameters
    ----------
    dirname : str or Path
        Path to the OLCI .SEN3 directory.
    chunks : int or tuple, optional
        Size of chunks for spatial dimensions. If int, applies to both dimensions. Default is 500.
    interp_angles : {'atan2', 'linear', 'legacy'}, optional
        Interpolation method for angles:
        - 'linear': Linear interpolation for all angles
        - 'atan2': Trigonometric interpolation (sin/cos then atan2)
        - 'legacy': Backward compatible (nearest for azimuth, linear for zenith).
        Default is 'linear'.
    metadata_template : list or None, optional
        List of metadata keys to include. If None, includes all metadata.
        Use empty list [] for minimal metadata. Default is None.
    v1_compat : bool, optional
        If True, formats output to match version 1 structure. Default is False.
    engine : str or None, optional
        xarray engine for opening NetCDF files. Default is None.
    verbose : bool, optional
        If True, print debug information. Default is True.
        
    Examples
    --------
    >>> ds = Level1_OLCI('S3A_OL_1_EFR____*.SEN3/', chunks=1000)
    """
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Folder does not exist'
    if (dirname/dirname.name).exists(): 
        dirname = (dirname/dirname.name)
    
    # Format chunks
    chunks = format_chunks(chunks)
    chunks = {'rows': chunks[str(names.rows)], 'columns': chunks[str(names.columns)]}
    
    # read manifest file for file names and footprint
    if verbose: log.debug('Read metadata')
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    manifest = read_xml(dirname/'xfdumanifest.xml')
    ds.attrs['metadata'] = filter_fn(manifest, metadata_template)
    
    # Add latlon footprint
    objs = manifest['metadataSection']['metadataObject']
    frameSet_obj = next(o for o in objs if o['attributes']['ID'] == 'measurementFrameSet')
    footprint = frameSet_obj['metadataWrap']['xmlData']['frameSet']['footPrint']
    idata = iter(footprint['posList'].split())
    footprint = [(float(v), float(idata.__next__())) for v in idata]
    lat,lon = zip(*footprint)
    ds.attrs['footprint_lat'] = lat
    ds.attrs['footprint_lon'] = lon
    
    # Get band informations
    bandnames, cwvl = [], []
    olciInfo_obj = next(o for o in objs if o['attributes']['ID'] == 'olciProductInformation')
    info = olciInfo_obj['metadataWrap']['xmlData']['olciProductInformation']
    for bn, data in info['bandDescriptions'].items():
        if bn == 'attributes': continue
        bandnames.append(bn)
        cwvl.append(data['centralWavelength'])
    ds = ds.assign_coords({str(names.bands): bandnames})
    ds = ds.assign({str(names.cwav): ((str(names.bands)),cwvl)})
    ds = ds.assign_coords({str(names.bgroup): (str(names.bands), ['bands_vnir']*len(cwvl))})
    
    # Check if product level is 1
    text = manifest['informationPackageMap']['contentUnit']['attributes']
    levels = findall(r'Level .', text['textInfo'])
    assert len(levels) == 1, f'Invalid textinfo in manifest: {text["textInfo"]}'
    level_from_manifest = levels[0].replace('Level ','level')
    assert 'level1' == level_from_manifest, \
        f'expected level1 encountered {level_from_manifest}'

    # Read main product
    if verbose: log.debug('Read radiances')
    ds = _Internal.read_bands(ds, dirname, chunks, 1)

    # Geo coordinates
    geo_coords_file = dirname/'geo_coordinates.nc'
    geo = xr.open_dataset(geo_coords_file, engine=engine, chunks=chunks)
    for k in geo.variables: 
        ds[k] = geo[k].astype('float32')
        ds[k].attrs.update(geo.attrs)

    # tie geometry interpolation
    if verbose: log.debug('read geometric tie points')
    _Internal.read_angle(ds, dirname, interp_angles, chunks, engine)

    # tie meteo interpolation
    if verbose: log.debug('read meteorological tie points')
    _Internal.read_ancillary_data(ds, dirname, chunks, engine)

    # instrument data
    instrument_data = xr.open_dataset(
        dirname/'instrument_data.nc',
        engine=engine,
        mask_and_scale=False,
        chunks=chunks,
        # this variable has duplicate dimensions, drop it
        drop_variables='relative_spectral_covariance'
    )
    ds = ds.assign({x: instrument_data[x] for x in instrument_data.variables})

    # quality flags
    if verbose: log.debug('read quality masks')
    qf = xr.open_dataset(dirname/'qualityFlags.nc', engine=engine, chunks=chunks)
    for var in qf.variables: 
        ds[var] = qf[var]

    # dimensions
    ds = ds.rename({'rows':str(names.rows), 'columns':str(names.columns)})
    ds = ds.chunk(dict(detectors=-1))   # FIXME: do this upstream
    
    # attributes
    if verbose: log.debug('add important attributes')
    _Internal.add_attributes(ds, manifest, dirname)
    ds.attrs['_flag_reader'] = 'eoread.olci.FlagsReader_OLCI'

    # SRF getter
    ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_olci'
    ds.attrs['_srf_getter_arg'] = {
        'Sentinel-3A': 'sentinel3_1_olci',
        'Sentinel-3B': 'sentinel3_2_olci',
    }[ds.attrs[str(names.platform)]]
    
    if verbose: log.debug('compute reflectances')
    ds = _Internal.olci_init_spectral(ds)
    ds = _Internal.extract_rtoa(ds)

    if v1_compat: return _v1_compat(ds)
    return ds.unify_chunks()


def Level2_OLCI(
        dirname: Union[str, Path],
        chunks: Union[int, tuple] = 500,
        interp_angles: Literal['atan2', 'linear', 'legacy'] = 'linear',
        metadata_template: Union[list, None] = None, 
        engine: str | None = None,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a Sentinel-3 OLCI Level2 product as an xarray.Dataset.
    
    Processes Level2 water products including water-leaving reflectances,
    chlorophyll concentration, aerosol properties, and quality flags.

    Parameters
    ----------
    dirname : str or Path
        Path to the OLCI Level2 .SEN3 directory.
    chunks : int or tuple, optional
        Size of chunks for spatial dimensions. Default is 500.
    interp_angles : {'atan2', 'linear', 'legacy'}, optional
        Interpolation method for angles. Default is 'linear'.
    metadata_template : list or None, optional
        List of metadata keys to include. If None, includes all metadata. Default is None.
    engine : str or None, optional
        xarray engine for opening NetCDF files. Default is None.
    verbose : bool, optional
        If True, print debug information. Default is True.
    """
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Folder does not exist'
    if (dirname/dirname.name).exists():
        dirname = (dirname/dirname.name)
        
    # Format chunks
    chunks = format_chunks(chunks)
    chunks = {'rows': chunks[str(names.rows)], 'columns': chunks[str(names.columns)]}

    # read manifest file for file names and footprint
    if verbose: log.debug('Read metadata')
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    manifest = read_xml(dirname/'xfdumanifest.xml')
    ds.attrs['metadata'] = filter_fn(manifest, metadata_template)
    
    # Add latlon footprint
    objs = manifest['metadataSection']['metadataObject']
    frameSet_obj = next(o for o in objs if o['attributes']['ID'] == 'measurementFrameSet')
    footprint = frameSet_obj['metadataWrap']['xmlData']['frameSet']['footPrint']
    idata = iter(footprint['posList'].split())
    footprint = [(float(v), float(idata.__next__())) for v in idata]
    lat,lon = zip(*footprint)
    ds.attrs['footprint_lat'] = lat
    ds.attrs['footprint_lon'] = lon
    
    # Get band informations
    bandnames, cwvl = [], []
    olciInfo_obj = next(o for o in objs if o['attributes']['ID'] == 'olciProductInformation')
    info = olciInfo_obj['metadataWrap']['xmlData']['olciProductInformation']
    for bn, data in info['bandDescriptions'].items():
        if bn == 'attributes': continue
        bandnames.append(bn)
        cwvl.append(data['centralWavelength'])
    ds = ds.assign_coords({str(names.bands): bandnames})
    ds = ds.assign({str(names.cwav): ((str(names.bands)),cwvl)})
    ds = ds.assign_coords({str(names.bgroup): (str(names.bands), ['bands_vnir']*len(cwvl))})
    
    # Retrieve product level
    text = manifest['informationPackageMap']['contentUnit']['attributes']
    levels = findall(r'Level .', text['textInfo'])
    assert len(levels) == 1, f'Invalid textinfo in manifest: {text["textInfo"]}'
    level_from_manifest = levels[0].replace('Level ','level')
    assert 'level2' == level_from_manifest, \
        f'expected level2 encountered {level_from_manifest}'

    # Read main product
    if verbose: log.debug('Read radiances')
    ds = _Internal.read_bands(ds, dirname, chunks, 2)

    # Geo coordinates
    geo_coords_file = dirname/'geo_coordinates.nc'
    geo = xr.open_dataset(geo_coords_file, engine=engine, chunks=chunks)
    for k in geo.variables: 
        ds[k] = geo[k].astype('float32')
        ds[k].attrs.update(geo.attrs)

    # tie geometry interpolation
    if verbose: log.debug('read geometric tie points')
    _Internal.read_angle(ds, dirname, interp_angles, chunks, engine)

    # tie meteo interpolation
    if verbose: log.debug('read meteorological tie points')
    _Internal.read_ancillary_data(ds, dirname, chunks, engine)

    # instrument data
    instrument_data = xr.open_dataset(
        dirname/'instrument_data.nc',
        engine=engine,
        mask_and_scale=False,
        chunks=chunks,
        # this variable has duplicate dimensions, drop it
        drop_variables='relative_spectral_covariance'
    )
    instrument_data = instrument_data.sel(bands=[int(b[2:])-1 for b in bandnames])
    ds = ds.assign({x: instrument_data[x] for x in instrument_data.variables})

    # chl_nn
    qf = xr.open_dataset(dirname/'chl_nn.nc', engine=engine, chunks=chunks)
    ds['chl_nn'] = qf.CHL_NN

    # chl_oc4me
    qf = xr.open_dataset(dirname/'chl_oc4me.nc', engine=engine, chunks=chunks)
    ds['chl_oc4me'] = qf.CHL_OC4ME

    # quality flags
    qf = xr.open_dataset(dirname/'wqsf.nc', engine=engine, chunks=chunks)
    ds['wqsf'] = qf.WQSF

    # aerosol properties
    qf = xr.open_dataset(dirname/'w_aer.nc', engine=engine, chunks=chunks)
    ds['A865'] = qf.A865
    ds['T865'] = qf.T865
    
    # dimensions
    ds = ds.rename({'rows':str(names.rows), 'columns':str(names.columns)})
    
   # attributes
    if verbose: log.debug('add important attributes')
    _Internal.add_attributes(ds, manifest, dirname)

    ds = ds.chunk(dict(detectors=-1))   # FIXME: do this upstream
    
    if verbose: log.debug('compute reflectances')
    ds = _Internal.olci_init_spectral(ds)

    return ds.unify_chunks()


def get_sample(level: int = 1) -> Path:
    return collect_sample(f'LEVEL{level}_OLCI', 'eumdac', 'SENTINEL-3-OLCI-FR', level)


class FlagsReader_OLCI(FlagsReaderBase):
    """
    Flags reader for OLCI (Sentinel-3) data.
    
    Provides access to OLCI quality flags for identifying invalid pixels
    and quality indicators.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return ['quality_flags']  # Use viewing zenith angle as reference
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return 'quality_flags'
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the OLCI dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            OLCI dataset containing quality_flags variable.
        flag_name : GenericFlags
            Standard flag identifier.
        """
        if flag_name == GenericFlags.LAND:
            # Combine land and fresh_inland_water flags
            land = getflag(ds['quality_flags'], 'land')
            fresh_inland_water = getflag(ds['quality_flags'], 'fresh_inland_water')
            return land & ~fresh_inland_water
        elif flag_name == GenericFlags.L1_INVALID:
            return getflag(ds['quality_flags'], 'invalid')
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")


################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def read_ltao(dirname: Path, band: str, chunks: dict, collec: dict, engine: str | None = None) -> None:
        """Read Level1 radiance data for a single band."""
        filename = dirname/f'{band}_radiance.nc'
        da = xr.open_dataarray(filename, engine=engine, chunks=chunks)
        collec['ltoa'].append(da)
    
    @staticmethod
    def read_rtoa(dirname: Path, band: str, chunks: dict, collec: dict, engine: str | None = None) -> None:
        """Read Level2 reflectance data and uncertainties for a single band."""
        filename = dirname/f'{band}_reflectance.nc'
        ds = xr.open_dataset(filename, engine=engine, chunks=chunks)
        collec['rtoa'].append(ds[f'{band}_reflectance'])
        collec['unc'].append(ds[f'{band}_reflectance_unc'])
    
    @staticmethod
    def read_bands(ds: xr.Dataset, dirname: Path, chunks: dict, level: int) -> xr.Dataset:
        """Read spectral band radiance or reflectance data from NetCDF files."""
        if level == 1:
            prod_list = {'ltoa': []}
            reader = _Internal.read_ltao
        else:
            prod_list = {'rtoa': [], 'unc': []}
            reader =_Internal.read_rtoa
            
        for band in ds[str(names.bands)].values:
            reader(dirname, band, chunks, prod_list)

        if level == 1: 
            ds[str(names.ltoa)] = xr.concat(prod_list['ltoa'], dim=str(names.bands))
            ds[str(names.ltoa)].attrs.update(unit='W/sr/m^2')
        else: 
            ds[str(names.rho_w)] = xr.concat(prod_list['rtoa'], dim=str(names.bands))
            ds['uncertainty'] = xr.concat(prod_list['unc'], dim=str(names.bands))
            ds[str(names.rho_w)].attrs.update(unit=None)
        
        return ds

    @staticmethod
    def olci_init_spectral(ds: xr.Dataset) -> xr.Dataset:
        """Broadcast spectral (detector-wise) data to the full image grid."""
        # wavelength
        ds[str(names.wav)] = xr.apply_ufunc(
            lambda l0, di: l0[:,0,0,di],
            ds.lambda0,  # (bands x detectors)
            ds.detector_index,   # (rows x columns)
            dask='parallelized',
            input_core_dims=[['detectors'], []],
            output_dtypes=[ds.lambda0.dtype],
        )
        ds[str(names.wav)].attrs.update(ds.lambda0.attrs)

        # solar flux
        ds[str(names.F0)] = xr.apply_ufunc(
            lambda sf, di: sf[:,0,0,di],
            ds.solar_flux,  # (bands x detectors)
            ds.detector_index,   # (rows x columns)
            dask='parallelized',
            input_core_dims=[['detectors'], []],
            output_dtypes=[ds.solar_flux.dtype],
        )
        ds[str(names.F0)].attrs.update(ds.solar_flux.attrs)
        
        return ds
    
    @staticmethod
    def extract_rtoa(ds: xr.Dataset) -> xr.Dataset:
        """Compute TOA reflectance from radiance using solar geometry."""
        mus = np.cos(np.radians(ds.sza))
        ds = ds.assign({
            str(names.rtoa): ((str(names.bands),str(names.rows),str(names.columns)), 
            (np.pi*ds[str(names.ltoa)]/(mus*ds[str(names.F0)])).data)
        })
        ds[str(names.rtoa)].attrs.update(unit=None)
        return ds

    @staticmethod
    def decompose_flags(value: int, flags: dict) -> list:
        """Return list of flag meanings for a given binary flag value."""
        return [m for (m, v) in flags.items() if (v & value != 0)]
    
    @staticmethod
    def read_angle(ds: xr.Dataset, dirname: str, interp_angles: str, chunks: dict, engine: str | None = None) -> None:
        """Read and interpolate viewing/solar angles from tie points to full grid."""
        
        # Open the dataset containing geometric information
        ac_factor = ds.latitude.ac_subsampling_factor
        al_factor = ds.latitude.al_subsampling_factor
        tie_geom_file = dirname/'tie_geometries.nc'
        tie_ds = xr.open_dataset(tie_geom_file, engine=engine, chunks=-1)
        tie_ds = tie_ds.assign_coords(
            tie_columns=np.arange(tie_ds.sizes['tie_columns'])*ac_factor,
            tie_rows=np.arange(tie_ds.sizes['tie_rows'])*al_factor,
        )
        
        # Check first and last items are the same 
        assert tie_ds.tie_columns[0] == ds['columns'][0]
        assert tie_ds.tie_columns[-1] == ds['columns'][-1]
        assert tie_ds.tie_rows[0] == ds['rows'][0]
        assert tie_ds.tie_rows[-1] == ds['rows'][-1]
        
        # Determine interpolation strategy
        if interp_angles == 'linear': interp_aa, interp_za = 'linear', 'linear'
        elif interp_angles == 'atan2': interp_aa, interp_za = 'atan2', 'atan2'
        elif interp_angles == 'legacy': interp_aa, interp_za = 'nearest', 'linear'
        else: raise ValueError(f'Invalid interp_angles "{interp_angles}"')
        
        # Iterate over all angles
        shape = ds.latitude.shape
        shape = dict(tie_rows=shape[0], tie_columns=shape[1])
        mapping = dict(tie_rows='rows', tie_columns='columns')
        tie_chunks = dict(tie_rows=chunks['rows'], tie_columns=chunks['columns'])
        for (ds_full, ds_tie, method) in [
                    (str(names.sza), 'SZA', interp_za),
                    (str(names.saa), 'SAA', interp_aa),
                    (str(names.vza), 'OZA', interp_za),
                    (str(names.vaa), 'OAA', interp_aa),
                ]:
            
            # Super-sample of the raster
            if method == 'atan2':
                _cos = spatial_resample(np.cos(np.radians(tie_ds[ds_tie])), shape, tie_chunks, 'linear')
                _sin = spatial_resample(np.sin(np.radians(tie_ds[ds_tie])), shape, tie_chunks, 'linear')
                raster = np.degrees(np.arctan2(_sin, _cos))
            else:
                raster = spatial_resample(tie_ds[ds_tie], shape, tie_chunks, method)
            
            # Add tie points            
            ds[ds_full] = raster.rename(mapping).transpose('rows', 'columns')
            ds[ds_full].attrs = tie_ds[ds_tie].attrs
            ds[ds_full+'_tie'] = tie_ds[ds_tie]
        
        # check subsampling factors
        assert ((ds.sizes['columns']-1) == ac_factor*(tie_ds.sizes['tie_columns']-1))
        assert ((ds.sizes['rows']-1) == al_factor*(tie_ds.sizes['tie_rows']-1))
    
    @staticmethod
    def read_ancillary_data(ds: xr.Dataset, dirname: str, chunks: dict, engine: str | None = None) -> None:
        """Read and interpolate meteorological tie-point data to full grid."""
        
        # Open the dataset containing meteorologic information
        ac_factor = ds.latitude.ac_subsampling_factor
        al_factor = ds.latitude.al_subsampling_factor
        tie_meteo_file = dirname/'tie_meteo.nc'
        tie = xr.open_dataset(tie_meteo_file, engine=engine, chunks=-1)
        tie = tie.assign_coords(
            tie_columns = np.arange(tie.sizes['tie_columns'])*ac_factor,
            tie_rows = np.arange(tie.sizes['tie_rows'])*al_factor,
        )
        
        # Check first and last items are the same 
        assert tie.tie_columns[0] == ds['columns'][0]
        assert tie.tie_columns[-1] == ds['columns'][-1]
        assert tie.tie_rows[0] == ds['rows'][0]
        assert tie.tie_rows[-1] == ds['rows'][-1]

        shape = ds.latitude.shape
        shape = dict(tie_rows=shape[0], tie_columns=shape[1])
        mapping = dict(tie_rows='rows', tie_columns='columns')
        tie_chunks = dict(tie_rows=chunks['rows'], tie_columns=chunks['columns'])
        wind0 = spatial_resample(tie.horizontal_wind.isel(wind_vectors=0), shape, tie_chunks, 'linear')
        wind1 = spatial_resample(tie.horizontal_wind.isel(wind_vectors=1), shape, tie_chunks, 'linear')
        
        wind = np.sqrt(wind0**2 + wind1**2)
        ds['horizontal_wind'] = wind.rename(mapping).transpose('rows', 'columns')
        ds['horizontal_wind'].attrs = tie['horizontal_wind'].attrs
        for var_from, var_to in [
            ('humidity', 'humidity'),
            ('sea_level_pressure', 'sea_level_pressure'),
            ('total_columnar_water_vapour', 'total_columnar_water_vapour'),
            ('total_ozone', 'total_column_ozone')
            ]:
            raster = spatial_resample(tie[var_from], shape, tie_chunks, 'linear')
            ds[var_to] = raster.rename(mapping).transpose('rows', 'columns')
            ds[var_to].attrs = tie[var_from].attrs
            ds[var_to + '_tie'] = tie[var_from]
    
    @staticmethod
    def add_attributes(ds: xr.Dataset, metadata: dict, dirname: str) -> None:
        """Extract and add product metadata attributes to the dataset."""
        meta = metadata['metadataSection']['metadataObject']
        acq_obj = next(o for o in meta if o['attributes']['ID'] == 'acquisitionPeriod')
        date = acq_obj['metadataWrap']['xmlData']['acquisitionPeriod']
        start = datetime.fromisoformat(date['startTime'])
        stop  = datetime.fromisoformat(date['stopTime'])
        ds.attrs[str(names.datetime)] = (start + (stop - start)/2.).isoformat()
        
        plat_obj = next(o for o in meta if o['attributes']['ID'] == 'platform')
        platform = plat_obj['metadataWrap']['xmlData']['platform']
        ds.attrs[str(names.platform)] = platform['familyName'] + platform['number']
        ds.attrs[str(names.resolution)] = 500
        ds.attrs[str(names.sensor)] = platform['instrument']['familyName']['attributes']['abbreviation']
        ds.attrs[str(names.product_name)] = dirname.name
        ds.attrs[str(names.input_directory)] = str(dirname.parent)


def _v1_compat(ds: xr.Dataset) -> xr.Dataset:
    """Transform dataset to version 1 format for backward compatibility."""
    # Reset band coordinates
    ds = ds.assign_coords(bands=[400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709, 754, 760, 764, 767, 779, 865, 885, 900, 940, 1020]) 
    
    # rename bands variable
    ds = ds.assign(
        {
            str(names.rtoa): (
                (str(names.bands), str(names.rows), str(names.columns)),
                ds[str(names.rtoa)].data,
            )
        }
    )
    
    # Add flags
    ds[str(names.flags)] = xr.zeros_like(
        ds.vza,
        dtype=names.flags.dtype)
    qf = getflags(ds.quality_flags)

    # raise LAND mask when land is raised but not fresh_inland_water
    from .eo import raiseflag
    raiseflag(
        ds[str(names.flags)],
        "LAND", 1,
        ds.quality_flags & (qf["land"] + qf["fresh_inland_water"]) == qf["land"],
    )
    raiseflag(
        ds[str(names.flags)],
        "L1_INVALID", 4,
        ds.quality_flags & qf["invalid"],
    )
    
    # Complete attributes
    for k,v in list(ds.longitude.attrs.items())[5:]: ds.attrs[k] = v
    
    return ds