#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MERIS level1 reader

l1 = Level1_MERIS('MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1')
l1 = Level1_MERIS('ENV_ME_1_RRG____20071102T084033_..._DSI_R_NT____.SEN3')  # 4th reprocessing
"""


from datetime import datetime
from pathlib import Path
from threading import Lock
from warnings import warn
from dask.array import array

import epr
import numpy as np
import pandas as pd
import xarray as xr

from core import log
from core.geo.naming import names
from core.tools import merge, drop_unused_dims

from eoread.flags import GenericFlags, FlagsReaderBase
from eoread.olci import read_level1_safe
from eoread.tools import filter_metadata, collect_sample, format_chunks, sun_earth_distance


user_guide = 'https://archive.org/details/manualzilla-id-5933919/page/n13/mode/2up'

def Level1_MERIS(
        filepath: str|Path,
        dir_smile: str|Path = None,
        read_auxdata: bool | None = None,
        chunks: int|tuple = 500,
        metadata_template: list = None,
        v1_compat: bool = False,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read an MERIS Level1 product as an xarray.Dataset.

    Envisat N1 files are read with Level1_MERIS_ENVISAT, products of the
    4th reprocessing (SAFE format, .SEN3 directory) with Level1_MERIS_SAFE.

    See these two readers for the other parameters.
    """
    
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    if filepath.is_dir():
        assert not v1_compat, 'v1_compat is not supported for the SAFE format'
        return Level1_MERIS_SAFE(
            filepath,
            chunks=chunks,
            metadata_template=metadata_template,
            verbose=verbose,
        )
    return Level1_MERIS_ENVISAT(
        filepath,
        dir_smile=dir_smile,
        read_auxdata=read_auxdata,
        chunks=chunks,
        metadata_template=metadata_template,
        v1_compat=v1_compat,
        verbose=verbose,
    )


def Level1_MERIS_ENVISAT(
        filepath: str|Path,
        dir_smile: str|Path = None,
        read_auxdata: bool | None = None,
        chunks: int|tuple = 500,
        metadata_template: list = None,
        v1_compat: bool = False,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a MERIS Level1 product of the Envisat N1 format as an xarray.Dataset.
    Formats the Dataset so that it contains the TOA radiances,
    the angles on the full grid, etc.

    Parameters
    ----------
    filepath : str or Path
        Path of the MERIS file path (ex: 'MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1').
    dir_smile : str or Path, optional
        Relative path to MERIS per-detector characterization. Default is '../auxdata/meris/'.
    read_auxdata : bool, optional
        Deprecated, ignored: the per-detector solar flux (F0) and wavelength
        (wav) from dir_smile are always read.
    chunks : int or tuple, optional
        Size of chunks for spatial axis. Default is 500.
    metadata_template : list, optional
        If None, add all metadata in output xarray.Dataset attributes else add only specified metadata. Default is None.
    v1_compat : bool, optional
        If True, format output xarray.Dataset such as version 1. Default is False.
    verbose : bool, optional
        If True, print debug information. Default is True.
    """
    
    filepath = Path(filepath)
    ds = xr.Dataset()
    bname = filepath.name
    
    # Format chunks
    chunks = format_chunks(chunks)

    # epr api is not thread safe: we have to use a lock for safe file access
    lock = Lock()

    prod = epr.Product(str(filepath))
    ds.attrs['totalwidth'] = prod.get_scene_width()
    ds.attrs['totalheight'] = prod.get_scene_height()
    
    # Read metadata
    if verbose: log.debug('read metadata')
    metadata = _Internal.read_metadata(ds, prod, metadata_template)
    nb_bands = metadata['NUM_BANDS']
    
    # Assign band names and wavelengths
    ds = ds.assign({str(names.cwav): ((str(names.bands)), metadata['BAND_WAVELEN']/1e3)})
    ds = ds.assign_coords({str(names.bands): list(range(1, nb_bands+1))})
    ds[str(names.bands)] = ds[str(names.bands)].astype(str)

    # read all rasters
    if verbose: log.debug('load raster')
    for name in prod.get_band_names():
        band = prod.get_band(name)
        ds[name] = xr.DataArray(
            array(_Internal.READ_MERIS(band, lock)),
            dims=(str(names.rows), str(names.columns)),
        ).chunk(chunks)
        ds[name].attrs['unit'] = band.unit
        ds[name].attrs['description'] = band.description

    # L1_INVALID flag: COSMETIC, SUSPECT or INVALID bits (the bit positions are
    # resolved by epr from the product data dictionary)
    if verbose: log.debug('load l1_invalid')
    raster = epr.create_bitmask_raster(prod.get_scene_width(), prod.get_scene_height())
    with lock:
        prod.read_bitmask_raster(
            'l1_flags.COSMETIC OR l1_flags.SUSPECT OR l1_flags.INVALID', 0, 0, raster)
    ds['l1_invalid'] = xr.DataArray(
        raster.data.astype(bool),
        dims=(str(names.rows), str(names.columns)),
    ).chunk(chunks)
    ds['l1_invalid'].attrs.update(description='Invalid pixel (COSMETIC, SUSPECT or INVALID flag set)')

    # Rename several variables and compile radiance rasters
    if verbose: log.debug('concatenate ltoa rasters')
    ds = _Internal.rename_variables(ds)
    ds = merge(ds, dim=str(names.bands), dtype=str)
    ds = ds.chunk({str(names.bands):1})
    ds[str(names.ltoa)].attrs.update(units='mW/(m^2*sr*nm)')
    ds = _Internal.standard_meteo(ds)

    if verbose: log.debug('read auxilary data')
    if dir_smile is None: dir_smile = Path(__file__).parent/'auxdata'/'meris'
    else: dir_smile = Path(dir_smile)
    assert dir_smile.exists(), f'{dir_smile} does not exists'
    
    if bname.startswith('MER_RR'): res = 'rr'
    elif bname.startswith('MER_FR'): res = 'fr'
    else: raise Exception(f'Error, could not identify whether MERIS file is RR or FR ({bname})')

    if read_auxdata is not None:
        warn('read_auxdata is deprecated: F0 and wav are always read', DeprecationWarning)

    # Read date
    dstart = _Internal.read_date(metadata['SENSING_START'])
    dstop = _Internal.read_date(metadata['SENSING_STOP'])
    d = dstart + (dstop - dstart)//2
    ds.attrs[str(names.datetime)] = d.isoformat()

    # Per-detector solar flux (corrected for the sun-earth distance) and wavelength
    F0 = _Internal.read_detector_table(dir_smile/f'sun_spectral_flux_{res}.txt', 'E0', ds)
    F0 = F0/sun_earth_distance(d)**2
    ds[str(names.F0)] = _Internal.detector_lookup(F0, ds.detector_index)
    ds[str(names.F0)].attrs.update(
        units='mW/(m^2*nm)',
        description='Solar flux at the acquisition date (corrected for the sun-earth distance)')

    wav = _Internal.read_detector_table(dir_smile/f'central_wavelen_{res}.txt', 'lam', ds)
    ds[str(names.wav)] = _Internal.detector_lookup(wav, ds.detector_index)
    ds[str(names.wav)].attrs.update(unit='nm')

    # Read attributes
    ds.attrs[str(names.platform)] = 'ENVISAT'
    ds.attrs[str(names.sensor)] = 'MERIS'
    ds.attrs[str(names.resolution)] = 300
    ds.attrs[str(names.product_name)] = metadata['PRODUCT'].decode()
    ds.attrs[str(names.input_directory)] = str(filepath.parent)
    ds.attrs['user_guide'] = user_guide
    ds.attrs['_flag_reader'] = 'eoread.meris.FlagsReader_MERIS'

    # SRF getter
    ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_meris'
    ds.attrs['_srf_getter_arg'] = ''

    # Add band groups
    ds = drop_unused_dims(ds)
    ds = ds.assign_coords({str(names.bgroup): (str(names.bands), ['bands_vnir']*nb_bands)})
    
    if v1_compat: return _v1_compat(ds, prod, lock, chunks)
    return ds.unify_chunks()


def Level1_MERIS_SAFE(
        dirname: str|Path,
        chunks: int|tuple = 500,
        interp_angles: str = 'linear',
        metadata_template: list = None,
        engine: str | None = None,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a MERIS Level1 product of the 4th reprocessing (SAFE format, as
    OLCI) as an xarray.Dataset.

    Parameters
    ----------
    dirname : str or Path
        Path to the .SEN3 directory
        (ex: 'ENV_ME_1_RRG____20071102T084033_..._DSI_R_NT____.SEN3').

    See eoread.olci.Level1_OLCI for the other parameters.
    """
    dirname = Path(dirname)
    ds = read_level1_safe(
        dirname,
        'merisProductInformation',
        chunks=chunks,
        interp_angles=interp_angles,
        metadata_template=metadata_template,
        engine=engine,
        verbose=verbose,
    )
    ds[str(names.ltoa)].attrs = {'units': 'mW/(m^2*sr*nm)'}

    # Per-detector wavelength and solar flux (the solar flux of the product
    # is already corrected for the sun-earth distance)
    ds[str(names.wav)] = _Internal.detector_lookup(ds.lambda0, ds.detector_index)
    ds[str(names.wav)].attrs.update(units='nm')
    ds[str(names.F0)] = _Internal.detector_lookup(ds.solar_flux, ds.detector_index)
    ds[str(names.F0)].attrs.update(
        units='mW/(m^2*nm)',
        description='Solar flux at the acquisition date (corrected for the sun-earth distance)')

    # Attributes
    ds.attrs[str(names.platform)] = 'ENVISAT'
    ds.attrs[str(names.resolution)] = 1200 if '_RR' in ds.attrs[str(names.product_name)] else 300
    ds.attrs['user_guide'] = user_guide
    # same quality flags as OLCI
    ds.attrs['_flag_reader'] = 'eoread.olci.FlagsReader_OLCI'

    # SRF getter
    ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_meris'
    ds.attrs['_srf_getter_arg'] = ''

    return ds.unify_chunks()


def get_sample(level: int=1) -> Path:
    return collect_sample(f'LEVEL{level}_MERIS', None)


class FlagsReader_MERIS(FlagsReaderBase):
    """
    Flags reader for MERIS (Medium Resolution Imaging Spectrometer) data.
    
    Provides access to MERIS L1 flags for identifying invalid pixels and
    other quality indicators.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return ['l1_invalid']
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return 'l1_invalid'

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the MERIS dataset.

        Parameters
        ----------
        ds : xr.Dataset
            MERIS dataset containing the l1_invalid variable.
        flag_name : GenericFlags
            Standard flag identifier (currently only L1_INVALID supported).
        """
        if flag_name == GenericFlags.L1_INVALID:
            return ds['l1_invalid']
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")


################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def rename_variables(ds: xr.Dataset) -> xr.Dataset:
        """Rename MERIS variables to standardized naming conventions."""
        ds = ds.rename({v: v.replace('radiance', str(names.ltoa)) 
                        for v in ds.variables if 'radiance' in v})
        return ds.rename({
            'latitude'    : str(names.lat),
            'longitude'   : str(names.lon),
            'view_zenith' : str(names.vza), 
            'view_azimuth': str(names.vaa),
            'sun_zenith'  : str(names.sza),
            'sun_azimuth' : str(names.saa), 
        })

    @staticmethod
    def read_metadata(ds: xr.Dataset, product, template: list) -> dict:
        """Extract metadata from MERIS product header."""
        metadata = {}
        for ph in [product.get_mph(), product.get_sph()]:
            metadata.update({
                f.get_name(): f.get_elem(0) if f.get_num_elems() == 1 else f.get_elems()
                for f in ph.fields()})
        
        filter_fn = (lambda x,y: x) if template is None else filter_metadata 
        ds.attrs['metadata'] = filter_fn(metadata, template)   
        
        return metadata

    @staticmethod
    def standard_meteo(ds: xr.Dataset) -> xr.Dataset:
        """
        Add the embedded meteo data with standard names and units
        """
        ds['horizontal_wind'] = np.sqrt(ds.zonal_wind**2 + ds.merid_wind**2)
        ds['horizontal_wind'].attrs.update(units='m/s', description='Wind speed')
        ds['sea_level_pressure'] = ds.atm_press.assign_attrs(units='hPa')
        ds['total_column_ozone'] = ds.ozone.assign_attrs(units='Dobson')
        return ds

    @staticmethod
    def read_detector_table(filename: Path, prefix: str, ds: xr.Dataset) -> xr.DataArray:
        """
        Read a per-detector table (columns `{prefix}_band0`... for bands 1...)
        as a DataArray (bands, detectors)
        """
        df = pd.read_csv(filename, delimiter='\t')
        nbands = len(ds[str(names.bands)])
        assert len(df.columns) == nbands + 1
        return xr.DataArray(
            np.stack([df[f'{prefix}_band{i}'].values for i in range(nbands)]).astype('float32'),
            dims=(str(names.bands), 'detectors'),
            coords={str(names.bands): ds[str(names.bands)].values},
        )

    @staticmethod
    def detector_lookup(table: xr.DataArray, detector_index: xr.DataArray) -> xr.DataArray:
        """
        Broadcast a per-detector table (bands, detectors) to the image grid,
        using the detector index (NaN where the detector index is invalid)
        """
        return xr.apply_ufunc(
            lambda t, di: np.where(di >= 0, t[:, 0, 0, np.maximum(di, 0)], np.nan).astype(t.dtype),
            table,
            detector_index,
            dask='parallelized',
            input_core_dims=[['detectors'], []],
            output_dtypes=[table.dtype],
        )

    @staticmethod
    def read_date(dat: str):
        """Parse MERIS-formatted date string to Python datetime."""
        dat = dat.decode('utf-8')
        months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
        for i,m in enumerate(months): dat = dat.replace(f'-{m}-', f'-{i+1:02d}-')
        return datetime.strptime(dat, '%d-%m-%Y %H:%M:%S.%f')


    # FIXME : Following classes should be revized
    class READ_MERIS:
        """
        An array-like to read data from a given MERIS band.
        """
        def __init__(self, band, lock):
            width = band.product.get_scene_width()
            height = band.product.get_scene_height()
            self.band = band
            self.lock = lock
            self.shape = (height, width)
            self.dtype = {
                'float': np.float32,
                'short': np.int16,
                'uchar': np.uint8,
            }[epr.data_type_id_to_str(band.data_type)]
            self.ndim = len(self.shape)

        def __getitem__(self, keys):
            assert len(keys) == self.ndim
            start = []
            steps = []
            sizes = []
            sel = []
            for k in keys:
                if isinstance(k, slice):
                    st = k.start or 0
                    start.append(st)
                    steps.append(k.step or 1)
                    sizes.append(k.stop - st)
                    sel.append(slice(None))
                else:  # Indexing with int
                    start.append(0)
                    steps.append(1)
                    sizes.append(1)
                    sel.append(0)

            with self.lock:
                if (sizes[0] > steps[0]) and (sizes[1] > steps[1]):
                    r = self.band.read_as_array(
                        yoffset=start[0],
                        xoffset=start[1],
                        height=sizes[0],
                        width=sizes[1],
                        ystep=steps[0],
                        xstep=steps[1],
                    )
                else:
                    r = self.band.read_as_array(
                        yoffset=start[0],
                        xoffset=start[1],
                        height=sizes[0],
                        width=sizes[1],
                    )[::steps[0], ::steps[1]]
            assert r.dtype == self.dtype
            return r[sel[0], sel[1]]