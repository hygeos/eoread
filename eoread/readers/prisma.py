#!/usr/bin/env python3
"""
PRISMA Level-1 reader (L1 STD product).

    >>> from eoread.readers.prisma import Level1_PRISMA       # doctest: +SKIP
    >>> ds = Level1_PRISMA('PRS_L1_STD_OFFL_..._0001.he5')    # doctest: +SKIP

PRISMA is ASI's hyperspectral imaging spectrometer (VNIR ~400-1000 nm, SWIR
~950-2500 nm, nominal 30 m). The L1 STD product is an HDF-EOS file holding
the two TOA radiance cubes (uint16, scaled by the per-cube global
ScaleFactor/Offset attributes) on a 1000 x 1000 pixel grid, with per-pixel
geolocation, land-cover/cloud masks and a per-band saturation/corruption
matrix.

File layout:
    root attrs                  Product_ID ('PRS_L1_STD'), List_Cw_Vnir/Swir
                                (descending wavelength, 0 = invalid band),
                                List_Fwhm_*, ScaleFactor_*/Offset_*,
                                Product_StartTime (ISO-8601)
    HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/
                                VNIR_Cube / SWIR_Cube (rows, bands, columns)
                                uint16; *_PIXEL_SAT_ERR_MATRIX; LandCover_Mask;
                                Cloud_Mask
    HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/
                                Latitude_VNIR / Longitude_VNIR (the SWIR grids
                                are identical to the byte)
    <sibling L2C> HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/
                                Solar_Zenith_Angle, Observing_Angle,
                                Rel_Azimuth_Angle  (required companion: the
                                L1 only carries global sun angles)

Documented deviations from the v1 implementation (see REFACTORING.md):
- Canonical orientation: per-file flip to south-up / west-to-east (the sample
  is stored row 0 = southern edge, column 0 = eastern edge, so v2 column j
  carries raw column nx-1-j); v1 kept the native layout.
- Per-band chunking where v1 read one all-bands block per cube.
- `shortname` (from the product's `Product_ID` attribute) and `crs` attributes
  added (schema requirements; v1 did not set them).
- The `cwav` unit attribute is spelled `units` (v1 used `unit`).
- A missing L2C companion raises `ProductError` (v1 raised `AssertionError`).

The .he5 container is HDF-EOS, which the xarray NetCDF engines cannot open, so
the reader reads through h5py with a lazy per-chunk wrapper: the file is
re-opened on each chunk read, keeping the dask graph free of long-lived
handles (and picklable for process-based schedulers).
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import xarray as xr
import dask.array as da
from core.geo.naming import names
from core import log

from eoread.flags import GenericFlags
from eoread.reader import ProductError, Reader
from eoread.tools import filter_metadata


user_guide = 'http://prisma.asi.it/missionselect/docs/PRISMA%20Product%20Specifications_Is2_3.pdf'

# Fill value for invalid pixels in the raw radiance cubes (PRISMA ATBD)
INVALID_RAW = 65535

# Product groups
_HCO_DATA = 'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields'
_HCO_GEO = 'HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields'
_L2C_GEOM = 'HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields'


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
        chunks: tuple[int, ...] | None = None
    ) -> da.Array:
    """
    Create a lazy dask array over an HDF5 dataset.

    The *h5file* handle is only used to probe the dataset; the data itself is
    read through _H5Slice, which re-opens the file on each chunk read. With no
    explicit chunks the whole dataset is a single dask block: the product
    stores the radiance cubes as one compressed chunk (the entire dataset), so
    any finer block would pay the full-cube decompression anyway.
    """
    dset = h5file[name]
    assert isinstance(dset, h5py.Dataset), f'Expected HDF5 dataset at /{name}'
    obj = _H5Slice(str(path), str(dset.name), tuple(dset.shape), dset.dtype)
    # dask.from_array is not fully type annotated: its chunks parameter is
    # inferred as 'str' from the 'auto' default, hence the cast.
    return da.from_array(obj, chunks=cast(Any, chunks or tuple(dset.shape)))


def _l2c_path(l1: Path) -> Path | None:
    """
    Derive the Level2C companion file from a Level1 product.

    Uses the standard offline naming convention
    (PRS_L1_STD_OFFL_...he5 -> PRS_L2C_STD_...he5), falling back to matching
    the acquisition timestamps embedded in the name. Returns None when no
    candidate exists on disk.
    """
    l1 = Path(l1)
    if l1.name.startswith('PRS_L1_STD_OFFL_'):
        cand = l1.with_name(l1.name.replace('PRS_L1_STD_OFFL_', 'PRS_L2C_STD_', 1))
        if cand.exists():
            return cand

    m = re.search(r'_(\d{14})_(\d{14})_', l1.name)
    if m:
        hits = sorted(l1.parent.glob(f'PRS_L2C_*_{m.group(1)}_*.he5'))
        if len(hits) == 1:
            return hits[0]

    return None


class Level1_PRISMA(Reader):
    """PRISMA Level-1 STD product (ASI, PRS_L1_STD).

    TOA radiance in both spectral ranges (VNIR + SWIR). Band labels are
    scientific indices (``B001``...) and the bands are sorted by ascending
    central wavelength; the two groups overlap (VNIR up to ~977 nm, SWIR from
    ~943 nm), so the stable sort interleaves them at the boundary (the
    per-band ``bgroup`` coordinate keeps track of the origin). ``cwav``/
    ``fwhm`` carry the actual values. The per-pixel
    geometric angles (sza, vza, raa) come from the mandatory sibling Level2C
    companion, auto-discovered by name: the L1 product only carries global
    sun angles.

    Flags (see ``getflag``): LAND = land-cover mask non-zero, CLOUD = cloud
    mask set, L1_INVALID = at least one band with invalid (NaN) radiance.
    """

    sensor = 'PRISMA'
    level = 'L1'
    chunks_default = 500
    payload_var = str(names.ltoa)
    required_vars = (str(names.sza), str(names.vza), str(names.raa),
                     'PIXEL_SAT_ERR_MATRIX')
    sample_products = ('PRISMA_L1',)
    flags = True

    #: Variant table for product types within the level. The root global
    #: attributes of the .he5 are bytes, so they are decoded before detection.
    VARIANTS = {
        'std': dict(detect=dict(field='Product_ID', value='PRS_L1_STD')),
    }

    def read(self, path: Path) -> xr.Dataset:
        path = Path(path)
        assert path.exists(), f'{path} does not exist'

        if self.verbose:
            log.debug(f'Reading PRISMA {self.level} product {path.name}')

        # Per-pixel geometric angles live in the sibling Level2C product; the
        # L1 only carries global sun angles. Required, as in v1.
        level2c = _l2c_path(path)
        if level2c is None:
            raise ProductError(
                f'{self.sensor} {self.level}: no L2C companion file found '
                f'next to {path.name}; the per-pixel geometric angles (sza, '
                f'vza, raa) live in the L2C product and are required')

        rows, cols, bands = str(names.rows), str(names.columns), str(names.bands)
        dims2 = (rows, cols)
        dims3 = (bands, *dims2)
        cy = self.chunks[rows]
        cx = self.chunks[cols]

        h1 = h5py.File(path, 'r')
        try:
            # Root global attributes (HDF-EOS); bytes decoded to str.
            metadata = {k: _h5text(v) if isinstance(v, (bytes, str)) else v
                        for k, v in h1.attrs.items()}
            # Loud on a non-L1-STD product (root global attribute).
            self.detect_variant(metadata)

            ds = xr.Dataset()

            if self.verbose:
                log.debug('Read radiance cubes')
            ltoa_list, sat_list, cwls, fwhms, bgroups = [], [], [], [], []
            for key in ('vnir', 'swir'):
                Key = key.upper()
                # As stored in the product: descending wavelength, 0 = invalid band
                cw = np.asarray(h1.attrs[f'List_Cw_{key.title()}'], dtype='float64')
                fwhm_arr = np.asarray(h1.attrs[f'List_Fwhm_{key.title()}'], dtype='float64')
                scale = _h5float(h1.attrs[f'ScaleFactor_{key.title()}'])
                offset = _h5float(h1.attrs[f'Offset_{key.title()}'])
                assert len(cw) == len(fwhm_arr), \
                    f'inconsistent band tables for {Key}: {len(cw)} cw vs {len(fwhm_arr)} fwhm'

                # Loud on a calibration table / cube shape mismatch.
                nband = h1[f'{_HCO_DATA}/{Key}_Cube'].shape[1]
                assert nband == len(cw), \
                    f'{Key}_Cube carries {nband} bands but the product lists {len(cw)}'

                # Output bands in ascending wavelength order, dropping the invalid ones
                idx = np.where(cw > 0)[0][::-1]
                assert idx.size, f'no valid {Key} band listed by the product'
                cwls.append(cw[idx].astype('float32'))
                fwhms.append(fwhm_arr[idx].astype('float32'))
                bgroups += [f'bands_{key}'] * idx.size

                # Radiance cube (rows, bands, columns) -> scale and mask invalid
                # pixels in the raw domain. One dask block per cube: the file
                # stores each cube as a single compressed chunk (the whole
                # dataset), so any finer dask block re-decompresses the entire
                # cube on every read.
                raw = _lazy_dset(path, h1, f'{_HCO_DATA}/{Key}_Cube')
                sel = raw[:, idx, :]
                ltoa = np.where(sel < INVALID_RAW, (sel - offset) / scale, np.nan)
                ltoa_list.append(
                    xr.DataArray(ltoa.transpose((1, 0, 2)).astype('float32'), dims=dims3))

                # Per-band saturation/corruption matrix (bit 1: saturation,
                # bit 2: corrupted frame); same single-chunk storage as the
                # cube, so one dask block per matrix.
                sat = _lazy_dset(path, h1, f'{_HCO_DATA}/{Key}_PIXEL_SAT_ERR_MATRIX')
                sat_list.append(
                    xr.DataArray(sat[:, idx, :].transpose((1, 0, 2)), dims=dims3))

            if self.verbose:
                log.debug('Read geolocation')
            for var, src in [(str(names.lat), 'Latitude_VNIR'),
                             (str(names.lon), 'Longitude_VNIR')]:
                arr = _lazy_dset(path, h1, f'{_HCO_GEO}/{src}', (cy, cx)).astype('float32')
                ds[var] = xr.DataArray(arr, dims=dims2)

            if self.verbose:
                log.debug('Read quality masks')
            for mask in ('LandCover_Mask', 'Cloud_Mask'):
                arr = _lazy_dset(path, h1, f'{_HCO_DATA}/{mask}', (cy, cx))
                ds[mask] = xr.DataArray(arr, dims=dims2)
        finally:
            h1.close()

        ds[str(names.ltoa)] = xr.concat(ltoa_list, dim=bands)
        ds[str(names.ltoa)].attrs['units'] = 'W/m^2/um/sr'
        ds['PIXEL_SAT_ERR_MATRIX'] = xr.concat(sat_list, dim=bands)

        if self.verbose:
            log.debug('Read geometric angles')
        h2 = h5py.File(level2c, 'r')
        try:
            for var, src in [(str(names.sza), 'Solar_Zenith_Angle'),
                             (str(names.vza), 'Observing_Angle'),
                             (str(names.raa), 'Rel_Azimuth_Angle')]:
                arr = _lazy_dset(level2c, h2, f'{_L2C_GEOM}/{src}', (cy, cx)).astype('float32')
                ds[var] = xr.DataArray(arr, dims=dims2)
        finally:
            h2.close()

        # Spectral coordinates: scientific index labels; bands sorted by
        # ascending central wavelength. The VNIR/SWIR groups overlap in
        # wavelength, so the stable merge-sort interleaves them at the
        # boundary (the per-band bgroup coordinate keeps track of origin).
        cw = np.concatenate(cwls)
        order = np.argsort(cw, kind='stable')
        ds = ds.isel({bands: order})
        ds = ds.assign_coords(
            {bands: [f'B{i + 1:03d}' for i in range(cw.size)]})
        ds[str(names.cwav)] = xr.DataArray(
            cw[order], dims=(bands,),
            attrs={'units': 'nm', 'desc': 'Central wavelength'})
        ds['fwhm'] = xr.DataArray(
            np.concatenate(fwhms)[order], dims=(bands,),
            attrs={'units': 'nm', 'description': 'Full width at half maximum'})
        ds = ds.assign_coords({str(names.bgroup): (bands, np.array(bgroups)[order])})

        # Per-band chunking before the orientation flip: dask-backed variables
        # handle the negative-step isel transparently.
        ds = ds.chunk({bands: 1, rows: cy, cols: cx})

        # Canonical orientation (decided per file, as for HICO/ECOSTRESS).
        ds = self._orient(ds)

        try:
            start = datetime.fromisoformat(metadata['Product_StartTime'])
        except (KeyError, ValueError) as exc:
            raise ProductError(
                f'{self.sensor} {self.level}: cannot parse the product '
                f'acquisition time from Product_StartTime '
                f'{metadata.get("Product_StartTime")!r}') from exc

        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): 'PRISMA',
            str(names.product_name): path.name,
            str(names.input_directory): str(path.parent),
            str(names.datetime): start.isoformat(),
            # Nominal 30 m pixel size (ATBD); the sample grid measures
            # ~30 x ~32 m (off-nadir tilt). v1 hardcoded the same value.
            str(names.resolution): 30,
            str(names.shortname): metadata['Product_ID'],
            str(names.crs): 'EPSG:4326',
            'user_guide': user_guide,
            'metadata': metadata if self.metadata_template is None
                        else filter_metadata(metadata, self.metadata_template),
        })

        return self.finalize(ds)

    # ---------------------------------------------------------- orientation

    @staticmethod
    def _orient(ds: xr.Dataset) -> xr.Dataset:
        """Flip to the canonical south-up / west-to-east orientation, per file.

        The stored row order is the scan order; which edge is north depends on
        the orbit, so the flip is decided from a two-pixel latitude/longitude
        gradient check at the grid center (the swath is a tilted parallelogram,
        so corner pixels are not reliable).
        """
        rows, cols = str(names.rows), str(names.columns)
        ny, nx = ds.sizes[rows], ds.sizes[cols]
        cy, cx = ny // 2, nx // 2
        lat_first = float(ds[str(names.lat)].isel({rows: 0, cols: cx}).values)
        lat_last = float(ds[str(names.lat)].isel({rows: -1, cols: cx}).values)
        lon_first = float(ds[str(names.lon)].isel({rows: cy, cols: 0}).values)
        lon_last = float(ds[str(names.lon)].isel({rows: cy, cols: -1}).values)
        if lat_first > lat_last:
            ds = ds.isel({rows: slice(None, None, -1)})
        if lon_first > lon_last:
            ds = ds.isel({cols: slice(None, None, -1)})
        ny, nx = ds.sizes[rows], ds.sizes[cols]
        return ds.assign_coords({rows: np.arange(ny), cols: np.arange(nx)})

    # ---------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return ['LandCover_Mask', 'Cloud_Mask', str(names.ltoa)]

    @classmethod
    def flag_dims_like(cls) -> str:
        # Per-pixel flags are 2D (a 3D reference such as Ltoa would give
        # per-band flags, which is not what downstream processors expect)
        return str(names.lat)

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.LAND:
            return ds['LandCover_Mask'] > 0
        if flag_name == GenericFlags.CLOUD:
            return ds['Cloud_Mask'] == 1
        if flag_name == GenericFlags.L1_INVALID:
            return ds[str(names.ltoa)].isnull().any(dim=str(names.bands))
        raise ValueError(
            f'{cls.sensor} {cls.level}: unsupported flag {flag_name!r}')
