#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

import dask.array as da
import pandas as pd
import xarray as xr

from eoread.flags import GenericFlags, FlagsReaderBase
from eoread.tools import (
    spatial_resample, 
    filter_metadata,
    format_chunks,
    collect_sample
)

from core.geo.naming import names
from core.tools import merge, drop_unused_dims
from core import log


def get_sample(level: int=1) -> Path:
    """
    Bring a SGLI file path to test reading function

    Parameters
    ----------
    level : int, optional
        Level of the product. Default is 1.
    """
    return collect_sample(f'LEVEL{level}_SGLI', None)


sgli_central_wavelengths = da.array([
    380.00, 412.00, 443.00, 490.00,
    530.00, 565.00, 673.50, 673.50,
    763.00, 868.50, 868.50], dtype='float32')


def Level1_SGLI(
        filepath: str|Path,
        chunks: int|tuple = 500,
        metadata_template: list | None = None,
        add_ancillary_data: bool = False,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read an SGLI Level1 product as an xarray.Dataset.
    Formats the Dataset so that it contains the TOA radiances,
    the angles on the full grid, etc.

    Parameters
    ----------
    filepath : str or Path
        Path of the ECOSTRESS H5file (Ex: GC1SG1_201912050000N02307_1BSG_VNRDK_1007.h5).
    chunks : int or tuple, optional
        Size of chunks for spatial axis. Default is 500.
    metadata_template : list or None, optional
        If None, add all metadata in output xarray.Dataset attributes else add only specified metadata. Default is None.
    add_ancillary_data : bool, optional
        If True, add ancillary data contained in provided file to the output dataset. Default is False.
    verbose : bool, optional
        If True, print debug information. Default is True.
    """
    
    ds = xr.Dataset()
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    
    # Format chunks
    chunks = format_chunks(chunks)

    # open image_data
    tree = xr.open_datatree(filepath, engine='h5netcdf', phony_dims='sort')
    imdata = tree['Image_data'].to_dataset()
    
    # read metadata
    if verbose: log.debug('Reading metadata files')
    metadata = _Internal.read_metadata(ds, tree, metadata_template)
    ds = ds.assign_coords({str(names.bands): metadata['Stored_channels'].split(',')})
    ds = ds.assign({str(names.cwav): ((str(names.bands)), sgli_central_wavelengths)})
    
    # Rename radiance dimensions
    imdata = imdata.rename_dims(dict(zip(
        imdata.Lt_VN01.dims,
        (str(names.rows), str(names.columns))
    )))
    sizes = imdata.Lt_VN01.sizes

    if verbose: log.debug('Read and compute geometric angles')
    _Internal.init_geometry(ds, tree, sizes, chunks)
    
    if verbose: log.debug('Read top of atmosphere data')
    ds = _Internal.read_toa(ds, imdata, chunks)
    ds = _Internal.read_mask(ds, imdata, chunks)

    # Attributes
    if verbose: log.debug('Add important attributes')
    ds.attrs[str(names.datetime)] = metadata['Scene_center_time']
    ds.attrs[str(names.product_name)] = metadata['Product_file_name']
    ds.attrs[str(names.platform)] = 'GCOM-C'
    ds.attrs[str(names.sensor)] = 'SGLI'
    ds.attrs[str(names.resolution)] = 250
    ds.attrs[str(names.input_directory)] = str(filepath.parent)
    ds.attrs['_flag_reader'] = 'eoread.sgli.FlagsReader_SGLI'

    # SRF getter
    ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_eumetsat'
    ds.attrs['_srf_getter_arg'] = 'gcom-c_1_sgli'
    
    if add_ancillary_data: 
        if verbose: log.info('Read ancillary data')
        ds = _Internal.read_ancillary(ds, tree)
    
    bgroup = ['bands'] * len(ds[str(names.bands)])
    ds = ds.assign_coords({str(names.bgroup): (str(names.bands), bgroup)})
    ds = ds.transpose(..., str(names.rows), str(names.columns))
    
    return ds.unify_chunks()


class FlagsReader_SGLI(FlagsReaderBase):
    """
    Flags reader for SGLI (Second-generation Global Imager) data from GCOM-C.
    
    Provides access to quality flags including land/water mask and
    quality indicators.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return ['quality_flag','water']  # Use viewing zenith angle as reference
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return 'quality_flag'
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the SGLI dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            SGLI dataset containing quality_flag and water variables.
        flag_name : GenericFlags
            Standard flag identifier (L1_INVALID or LAND).
        """
        if flag_name == GenericFlags.L1_INVALID:
            return ds['quality_flag']
        if flag_name == GenericFlags.LAND:
            return ~ds['water']
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")


################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def read_toa(ds: xr.Dataset, imdata: xr.Dataset, chunks: dict) -> xr.Dataset:
        """Read and calibrate TOA reflectance from SGLI image data."""
        
        mus = da.cos(da.radians(ds.sza))
        
        for band in ds.bands.values:
            Rtoa = imdata[f'Lt_{band[:4]}'].chunk(chunks)
            attrs = Rtoa.attrs
            Rtoa = (Rtoa & attrs['Mask']) * attrs['Slope_reflectance']
            Rtoa = (Rtoa + attrs['Offset_reflectance'])/mus
            Rtoa.attrs = attrs
            ds[str(names.rtoa)+f'_{band}'] = Rtoa
            
        pattern = f'({str(names.rtoa)})'+r'_(.+)'
        ds = merge(ds, dim=str(names.bands), pattern=pattern, dtype=str)
        ds[str(names.rtoa)].attrs['unit'] = None
        return ds

    @staticmethod
    def read_mask(ds: xr.Dataset, imdata: xr.Dataset, chunks: dict) -> xr.Dataset:
        """Read land/water mask and quality flags from SGLI image data."""
        ds['water'] = imdata['Land_water_flag'].chunk(chunks)
        ds['quality_flag'] = imdata['QA_flag'].chunk(chunks)
        return ds
    
    @staticmethod
    def init_geometry(ds: xr.Dataset, tree: xr.Dataset, shape: dict, chunks: dict) -> None:
        """Read and interpolate geometric angles from tie points to full resolution."""
        
        # Transform into dataset and rename axis
        geom = tree['Geometry_data'].to_dataset()
        geom = geom.rename_dims(dict(zip(
            geom.Latitude.dims,
            (str(names.rows)+'_tie', str(names.columns)+'_tie')
        )))
        
        # Add tie points into dataset
        ds['lat_tie'] = geom.Latitude
        ds['lon_tie'] = geom.Longitude
        ds['vza_tie'] = geom['Sensor_zenith']
        ds['vaa_tie'] = geom['Sensor_azimuth']
        ds['sza_tie'] = geom['Solar_zenith']
        ds['saa_tie'] = geom['Solar_azimuth']
        
        # Apply slope transformation
        delta = 10
        for x in [x for x in ds if x.endswith('_tie')]:
            assert ds[x].Resampling_interval == delta
            assert ds[x].Offset == 0.
            ds[x] = (1 + ds[x].Slope) * ds[x]

        # assign tiepoint coordinates
        mapping = {str(names.rows)+'_tie': str(names.rows), str(names.columns)+'_tie': str(names.columns)}
        ds[str(names.columns)+'_tie'] = da.arange(ds.sizes[str(names.columns)+'_tie'])*delta
        ds[str(names.rows)+'_tie'] = da.arange(ds.sizes[str(names.rows)+'_tie'])*delta

        # Create interpolated datasets
        for (name, A) in [
                (str(names.lat), ds.lat_tie),
                (str(names.lon), ds.lon_tie),
                (str(names.vza), ds.vza_tie),
                (str(names.vaa), ds.vaa_tie),
                (str(names.sza), ds.sza_tie),
                (str(names.saa), ds.saa_tie),
            ]:
            A = A.rename(mapping)
            ds[name] = spatial_resample(A, shape, chunks=chunks)

    @staticmethod
    def calc_central_wavelength():
        """Read SRF and calculate central wavelength for each band"""
        dir_auxdata = Path(__file__).parent/'auxdata'/'sgli'

        file_rsr = dir_auxdata/'sgli_rsr_f_for_algorithm_201008.txt.gz'
        assert file_rsr.exists(), file_rsr

        rsr = pd.read_csv(
            file_rsr,
            engine='python',
            delim_whitespace=True,
            index_col=False,
        )

        rsr = rsr.rename(columns=dict(zip(
            [x for x in rsr.columns if x.startswith('WL')],
            [x.replace('RSR_', 'WL_') for x in rsr.columns if x.startswith('RSR')],
        )))

        wav_data = []

        # calculate central wavelengths
        for i, _ in enumerate(sgli_bands):
            srf = rsr[f'RSR_VN{i+1:02}']
            wav = rsr[f'WL_VN{i+1:02}']
            wav_eq = da.trapz(wav*srf)/da.trapz(srf)
            wav_data.append(wav_eq)

        return sgli_bands, wav_data

    @staticmethod
    def read_metadata(ds: xr.Dataset, tree: dict, template: list) -> dict:
        """Extract global attributes from SGLI HDF5 file."""
        filter_fn = (lambda x,y: x) if template is None else filter_metadata
        metadata = tree['Global_attributes'].attrs
        ds.attrs['metadata'] = filter_fn(metadata, template)
        return metadata

    @staticmethod
    def read_ancillary(ds: xr.Dataset, tree: dict) -> xr.Dataset:
        """Read ancillary data from SGLI HDF5 file."""
        ancillary = tree['Ancillary_data'].to_dict()
        for name, data in ancillary.items():
            for var, val in data.variables.items():
                n = '/'.join([name, var]) 
                ds = ds.assign({n:val})
        return ds