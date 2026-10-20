#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRISMA Level1 reader

References:
    http://prisma.asi.it/missionselect/docs/PRISMA%20ATBD_v1.pdf
    http://prisma.asi.it/missionselect/docs/PRISMA%20Product%20Specifications_Is2_3.pdf
"""

import re

import h5py
import numpy as np
import xarray as xr
import dask.array as da

from pathlib import Path
from typing import Any, cast

from core.geo.naming import names
from core.tools import drop_unused_dims
from core import log

from eoread.flags import FlagsReaderBase, GenericFlags
from eoread.tools import (
    collect_sample,
    filter_metadata,
    format_chunks,
)


user_guide = 'http://prisma.asi.it/missionselect/docs/PRISMA%20Product%20Specifications_Is2_3.pdf'

# Fill value used for invalid pixels in the raw radiance cubes
INVALID_RAW = 65535


def _h5text(value) -> str:
    """Decode an HDF5 string attribute (bytes or str depending on encoding)."""
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else str(value)


def _h5float(value) -> float:
    """Coerce a scalar HDF5 attribute value to float."""
    return float(np.asarray(value).item())


class _H5Slice:
    """
    Lazy wrapper over an h5py dataset.

    The HDF5 file is opened on each chunk read, so the dask graph holds no
    long-lived file handle and the object remains picklable (process-based
    schedulers work).
    """

    def __init__(self, path: str, h5path: str, shape: tuple, dtype):
        self.path = str(path)
        self.h5path = h5path
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, key):
        with h5py.File(self.path, 'r') as f:
            dset = f[self.h5path]
            assert isinstance(dset, h5py.Dataset)
            return dset[key]


def _lazy_dset(
        path: str | Path,
        h5file: h5py.File,
        name: str,
        chunks: tuple[int, ...]
    ) -> da.Array:
    """
    Create a lazy dask array over an HDF5 dataset.

    The *h5file* handle is only used to probe the dataset; the data itself is
    read through _H5Slice, which re-opens the file on each chunk read.
    """
    dset = h5file[name]
    assert isinstance(dset, h5py.Dataset), f'Expected HDF5 dataset at /{name}'
    obj = _H5Slice(str(path), str(dset.name), tuple(dset.shape), dset.dtype)
    # dask.from_array is not fully type annotated: its chunks parameter is
    # inferred as 'str' from the 'auto' default, hence the cast.
    return da.from_array(obj, chunks=cast(Any, chunks))


def _l2c_path(l1: Path) -> Path | None:
    """
    Derive the Level2C companion file from a Level1 product.

    Uses the standard offline naming convention
    (PRS_L1_STD_OFFL_...he5 -> PRS_L2C_STD_...he5), falling back to matching
    the acquisition timestamps embedded in the filename.

    Returns None if no unique companion file can be identified.
    """
    candidate = l1.with_name(l1.name.replace('PRS_L1_STD_OFFL_', 'PRS_L2C_STD_'))
    if candidate.exists():
        return candidate

    m = re.search(r'(\d{14}_\d{14})', l1.name)
    if not m:
        return None
    matches = list(l1.parent.glob(f'PRS_L2C_*_{m.group(1)}_*.he5'))
    if len(matches) != 1:
        return None
    return matches[0]


def Level1_PRISMA(
        filename: str | Path,
        chunks: int | tuple = 500,
        metadata_template: list | None = None,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a PRISMA Level1 product as an xarray Dataset.

    PRISMA provides VNIR (400-1000 nm) and SWIR (950-2500 nm) hyperspectral
    cubes at 30 m spatial resolution. Both cubes are exposed in a single
    dataset under the variable `Ltoa`, distinguished by the `bands_group`
    coordinate ('bands_vnir' / 'bands_swir'). Bands are stored in ascending
    wavelength order; invalid bands (central wavelength == 0) are dropped.
    Invalid pixels (raw fill value 65535) are set to NaN in `Ltoa`; the
    per-band saturation/corruption matrix is exposed as
    `PIXEL_SAT_ERR_MATRIX` (bit 1: saturation, bit 2: corrupted frame).

    The reader is lazy: all variables are backed by dask arrays and data is
    only read on demand, chunk-wise over the spatial dimensions.

    The Level2C companion product (same directory) is required to provide the
    geometric angles (sza, vza, raa).

    Args:
        filename: Path to the PRISMA Level1 .he5 file
        chunks: Size of chunks for spatial dimensions. If int, applies to both
                dimensions. If tuple/list, should be (rows_chunk, columns_chunk)
        metadata_template: List of metadata keys to include. If None, includes all metadata.

    Example:
        >>> ds = Level1_PRISMA('PRS_L1_STD_OFFL_20220423112851_20220423112856_0001.he5')
    """
    filename = Path(filename).resolve()
    assert filename.exists(), 'File does not exist'
    assert filename.name.startswith('PRS_L1_'), f'Expected a PRISMA Level1 product, got {filename.name}'

    level2c = _l2c_path(filename)
    assert level2c is not None and level2c.exists(), \
        f'L2C companion file is required for {filename.name}'

    chunk_sizes = format_chunks(chunks)
    cy = chunk_sizes[str(names.rows)]
    cx = chunk_sizes[str(names.columns)]
    dims2 = (str(names.rows), str(names.columns))
    dims3 = (str(names.bands), *dims2)

    h1 = h5py.File(filename, 'r')
    HCO = 'HDFEOS/SWATHS/PRS_L1_HCO'

    if verbose: log.debug('Read product metadata')
    ds = xr.Dataset()
    metadata = {k: _h5text(v) if isinstance(v, (bytes, str)) else v for k, v in h1.attrs.items()}
    if metadata_template is None:
        ds.attrs['metadata'] = metadata
    else:
        ds.attrs['metadata'] = filter_metadata(metadata, metadata_template)

    # Attributes
    ds.attrs[str(names.datetime)] = _h5text(h1.attrs['Product_StartTime'])
    ds.attrs[str(names.platform)] = 'PRISMA'
    ds.attrs[str(names.sensor)] = 'PRISMA'
    ds.attrs[str(names.resolution)] = 30
    ds.attrs[str(names.product_name)] = filename.name
    ds.attrs[str(names.input_directory)] = str(filename.parent)
    ds.attrs['user_guide'] = user_guide
    ds.attrs['_flag_reader'] = 'eoread.prisma.FlagsReader_PRISMA'

    if verbose: log.debug('Read radiance cubes')
    ltoa_list, sat_list, bnames, cwls, fwhms, bgroups = [], [], [], [], [], []
    for key in ['vnir', 'swir']:
        Key = key.upper()
        # As stored in the product: descending wavelength, 0 = invalid band
        cw = np.asarray(h1.attrs[f'List_Cw_{key.title()}'])
        fwhm = np.asarray(h1.attrs[f'List_Fwhm_{key.title()}'])
        scale = _h5float(h1.attrs[f'ScaleFactor_{key.title()}'])
        offset = _h5float(h1.attrs[f'Offset_{key.title()}'])

        # Output bands in ascending wavelength order, dropping the invalid ones
        idx = np.where(cw > 0)[0][::-1]
        cwls.append(cw[idx].astype('float32'))
        fwhms.append(fwhm[idx].astype('float32'))
        bnames += [str(int(c)) for c in cw[idx]]
        bgroups += [f'bands_{key}']*len(idx)

        # Radiance cube (rows, bands, columns) -> scale and mask invalid pixels in the raw domain
        raw = _lazy_dset(filename, h1, f'{HCO}/Data Fields/{Key}_Cube', (cy, -1, cx))
        sel = raw[:, idx, :]
        ltoa = np.where(sel < INVALID_RAW, (sel - offset)/scale, np.nan)
        ds_ltoa = xr.DataArray(ltoa.transpose((1, 0, 2)).astype('float32'), dims=dims3)
        ltoa_list.append(ds_ltoa)

        # Per-band saturation/corruption error matrix (bit 1: saturation, bit 2: corrupted frame)
        sat = _lazy_dset(filename, h1, f'{HCO}/Data Fields/{Key}_PIXEL_SAT_ERR_MATRIX', (cy, -1, cx))
        sat_list.append(xr.DataArray(sat[:, idx, :].transpose((1, 0, 2)), dims=dims3))

    ds[str(names.ltoa)] = xr.concat(ltoa_list, dim=str(names.bands))
    ds[str(names.ltoa)].attrs.update(units='W/m^2/um/sr')
    ds['PIXEL_SAT_ERR_MATRIX'] = xr.concat(sat_list, dim=str(names.bands))

    if verbose: log.debug('Read geolocation')
    for var, src in [(str(names.lat), 'Latitude_VNIR'), (str(names.lon), 'Longitude_VNIR')]:
        arr = _lazy_dset(filename, h1, f'{HCO}/Geolocation Fields/{src}', (cy, cx)).astype('float32')
        ds[var] = xr.DataArray(arr, dims=dims2)

    if verbose: log.debug('Read quality masks')
    for mask in ['LandCover_Mask', 'Cloud_Mask']:
        arr = _lazy_dset(filename, h1, f'{HCO}/Data Fields/{mask}', (cy, cx))
        ds[mask] = xr.DataArray(arr, dims=dims2)
    h1.close()

    if verbose: log.debug('Read geometric angles')
    h2 = h5py.File(level2c, 'r')
    GEOM_L2C = 'HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields'
    for var, src in [
        (str(names.sza), 'Solar_Zenith_Angle'),
        (str(names.vza), 'Observing_Angle'),
        (str(names.raa), 'Rel_Azimuth_Angle'),
    ]:
        arr = _lazy_dset(level2c, h2, f'{GEOM_L2C}/{src}', (cy, cx)).astype('float32')
        ds[var] = xr.DataArray(arr, dims=dims2)
    h2.close()

    # Band coordinates and central wavelengths
    ds = ds.assign({str(names.bgroup): (str(names.bands), np.array(bgroups))})
    ds = ds.assign_coords({str(names.bands): np.array(bnames)})
    ds[str(names.cwav)] = xr.DataArray(np.concatenate(cwls, axis=0), dims=(str(names.bands),))
    ds[str(names.cwav)].attrs.update(unit='nm')
    ds['fwhm'] = xr.DataArray(np.concatenate(fwhms, axis=0), dims=(str(names.bands),))
    ds['fwhm'].attrs.update(units='nm', description='Full width at half maximum')

    ds = drop_unused_dims(ds)
    ds = ds.set_coords(str(names.bgroup))
    return ds.unify_chunks()


def get_sample(level: int = 1) -> Path:
    """
    Retrieve a sample PRISMA Level1 product for testing.

    Requires the LEVEL{level}_PRISMA environment variable pointing to a
    PRISMA Level1 .he5 file (no public download provider is available).
    """
    return collect_sample(f'LEVEL{level}_PRISMA', None)


class FlagsReader_PRISMA(FlagsReaderBase):
    """
    Flags reader for PRISMA Level1 products.

    - LAND: pixel classified as land by the embedded LandCover_Mask (value > 0)
    - CLOUD: Cloud_Mask == 1
    - L1_INVALID: NaN radiance in any band

    Per-band saturation/corruption is available through the
    PIXEL_SAT_ERR_MATRIX variable.
    """

    def requires(self) -> list[str]:
        return ['LandCover_Mask', 'Cloud_Mask', str(names.ltoa)]

    def dims_like(self) -> str:
        return str(names.sza)

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.LAND:
            return ds['LandCover_Mask'] > 0
        if flag_name == GenericFlags.CLOUD:
            return ds['Cloud_Mask'] == 1
        if flag_name == GenericFlags.L1_INVALID:
            return ds[str(names.ltoa)].isnull().any(dim=str(names.bands))
        raise ValueError(f'Unsupported flag: {flag_name}')
