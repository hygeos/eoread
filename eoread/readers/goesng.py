#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOES-NG (GOES-R series ABI) level L1b multi-channel reader.

The product family is the CMS/ICARE "Emultic" composition of GOES-16 ABI
direct-readout radiances:

- ``Emultic1kmNC4_goes16_YYYYMMDDHHMM.nc``: 1 km visible channels
  ``VIS_004`` (0.47 um), ``VIS_008`` (0.86 um) and ``VIS_016`` (1.6 um);
- ``Emultic500mNC4_goes16_YYYYMMDDHHMM.nc``: the 0.64 um channel
  ``VIS_006`` at 500 m, box-averaged 2x2 onto the 1 km grid;
- ``GOESNG-0750.1km.{nc,hdf}``: the ICARE static grid (lat/lon, viewing
  angles and pixel area size) for the GOES-16 full disk;
- optionally, ``S_NWC_CMA_GOES16_globeE-NR_YYYYMMDDTHHMMSSZ.nc``: the NWC/GEO
  cloud mask at 2 km, repeated 2x2 (``cloudmask=True``).

Deviation from the legacy function (``eoread.legacy.goesng``):

- The legacy function is unusable as-is: it references undefined globals
  (``naming``, ``eo``, ``flags``, ``file_1km``) and never reaches a result.
  This reader is a fresh implementation of the same product family, verified
  against real CMS/ICARE files.
- Output is **south-up** (row 0 = southern edge, canonical orientation). The
  stored geostationary grid is north-up; all fields are flipped together once.
- Band labels are sensor-native strings (``VIS_004`` ...), with central
  wavelengths in ``cwav`` (nm); legacy used the four integer wavelengths as
  the band coordinate and a separate ``wav`` variable.
- Geometry, reflectance and pixel-area fields are float32 (legacy kept the
  files' float64); ``Rtoa`` is dimensionless and carries the product's
  ``units='1'``.
- The NWC/GEO cloud mask is an optional ``cloud`` layer (2D float32, NaN off
  disk) exposed to the flags framework as ``GenericFlags.CLOUD``; legacy tried
  to fold it into a packed ``flags`` variable and failed (undefined ``flags``).
- The legacy ``convert_auxfile`` option is dropped: a NetCDF auxiliary grid is
  read directly and an HDF4 one lazily through ``eoread.hdf4.load_hdf4``
  (no on-disk conversion side effect).
"""

import re
from pathlib import Path

import dask.array as da
import numpy as np
import pysolar.solar as pysol
import xarray as xr
from dateutil.parser import parse

from core import log
from core.geo.naming import names

from eoread.flags import GenericFlags
from eoread.reader import ProductError, Reader
from eoread.tools import filter_metadata

#: Band labels, in output order, and their central wavelengths (nm).
#: Values carried over from the legacy reader (ABI channels 1, 2, 3, 5).
BANDS = ('VIS_004', 'VIS_006', 'VIS_008', 'VIS_016')
#: Bands stored in the 1 km file (VIS_006 comes from the 500 m file).
BANDS_1KM = ('VIS_004', 'VIS_008', 'VIS_016')
CWAV = (470., 640., 865., 1610.)

#: 1 km file -> 500 m counterpart token.
TOKEN_1KM = 'Emultic1kmNC4'
TOKEN_500M = 'Emultic500mNC4'

#: Auxiliary grid file names, looked up next to the product.
AUX_NAMES = ('GOESNG-0750.1km.nc', 'GOESNG-0750.1km.hdf',
             'ANCILLARY/GOESNG-0750.1km.hdf', 'ANCILLARY/GOESNG-0750.1km.nc')

#: Auxiliary variables required by the reader.
AUX_VARS = ('Latitude', 'Longitude', 'View_Zenith', 'View_Azimuth',
            'Pixel_Area_Size')


def _scaled_reflectance(var: xr.DataArray) -> xr.DataArray:
    """File reflectance is in percent (scale_factor 0.01): -> [0, 1] float32."""
    return (var / 100.).astype('float32')


class Level1_GOESNG(Reader):
    """GOES-NG ABI level L1b reflectance (ICARE/CMS ``Emultic`` composition)."""

    sensor = 'ABI'
    level = 'L1'
    chunks_default = 1000
    payload_var = str(names.rtoa)
    sample_products = ('GOESNG_L1',)
    required_vars = (str(names.sza), str(names.saa), str(names.vza), str(names.vaa))
    flags = True

    def __init__(self, path, chunks=None, metadata_template=None, verbose=False,
                 auxfile=None, cloudmask=False):
        """
        Parameters
        ----------
        path : str or Path
            The 1 km ``Emultic1kmNC4_*`` file. The 500 m counterpart and the
            auxiliary grid are discovered next to it.
        auxfile : str or Path, optional
            ICARE static grid (``GOESNG-0750.1km.nc`` or ``.hdf``). Defaults to
            the first existing candidate next to the product.
        cloudmask : bool
            Attach the NWC/GEO cloud mask as a ``cloud`` layer (requires the
            ``S_NWC_CMA_*`` file next to the product).
        """
        self.auxfile = Path(auxfile) if auxfile is not None else None
        self.cloudmask = cloudmask
        super().__init__(path, chunks=chunks, metadata_template=metadata_template,
                         verbose=verbose)

    # ------------------------------------------------------------------ read

    def read(self, path: Path) -> xr.Dataset:
        # Do not resolve(): the staged sample is a directory of symlinks, and
        # the auxiliary grid deliberately lives next to the links, not next to
        # their (different) targets.
        path = Path(path).absolute()
        if not path.exists():
            raise ProductError(f'GOESNG L1: {path} does not exist')
        if TOKEN_1KM not in path.name:
            raise ProductError(
                f'GOESNG L1: expected a {TOKEN_1KM} product, got {path.name!r}')

        if self.verbose:
            log.debug(f'Reading GOESNG L1 product {path.name}')

        file_500m = path.with_name(path.name.replace(TOKEN_1KM, TOKEN_500M, 1))
        if not file_500m.exists():
            raise ProductError(
                f'GOESNG L1: the paired 500 m file {file_500m.name} is missing '
                f'next to {path.name}')
        auxfile = self._find_aux(path)

        rows, cols, bands = str(names.rows), str(names.columns), str(names.bands)
        cy = self.chunks[rows]
        cx = self.chunks[cols]

        # 1. Reflectances on the 1 km grid ------------------------------------
        ds_1km = xr.open_dataset(
            path, chunks={'ny1km': cy, 'nx1km': cx},
            drop_variables=['dtime'], engine='h5netcdf')
        missing = [b for b in BANDS_1KM if b not in ds_1km]
        if missing:
            raise ProductError(
                f'GOESNG L1: {path.name} lacks the expected bands {missing}')

        bands_data = {}
        for band in BANDS_1KM:
            bands_data[band] = _scaled_reflectance(
                ds_1km[band].rename({'ny1km': rows, 'nx1km': cols}))

        # 2. The 500 m 0.64 um channel, box-averaged 2x2 onto 1 km -----------
        ds_500m = xr.open_dataset(
            file_500m, chunks={'ny500m': 2 * cy, 'nx500m': 2 * cx},
            drop_variables=['dtime'], engine='h5netcdf')
        if 'VIS_006' not in ds_500m:
            raise ProductError(
                f'GOESNG L1: {file_500m.name} lacks the VIS_006 channel')
        vis_006 = ds_500m['VIS_006'].rename({'ny500m': rows, 'nx500m': cols})
        # Exactly the legacy 2x2 box average: sum the four strided phases and
        # divide by 4 (NaN propagates, as in the legacy sum).
        acc = 0.
        for i in (0, 1):
            for j in (0, 1):
                acc = acc + vis_006.isel({rows: slice(i, None, 2),
                                          cols: slice(j, None, 2)})
        # Align the averaged band's chunks with the 1 km channels so the
        # concatenation keeps a single chunk boundary per band.
        bands_data['VIS_006'] = _scaled_reflectance(acc / 4.).chunk(
            {rows: cy, cols: cx})

        # 3. Static grid: geolocation, viewing angles, pixel area --------------
        aux = self._open_aux(auxfile, cy, cx)
        aux_dims = aux['Latitude'].dims
        if len(aux_dims) != 2:
            raise ProductError(
                f'GOESNG L1: auxiliary Latitude must be 2D, got {aux_dims}')
        aux = aux.rename({aux_dims[0]: rows, aux_dims[1]: cols})

        ds = xr.Dataset({self.payload_var: xr.concat(
            [bands_data[b] for b in BANDS], dim=bands)})
        ds = ds.assign_coords({bands: list(BANDS)})
        ds[str(names.cwav)] = (bands, np.asarray(CWAV, dtype='float64'))
        ds[str(names.cwav)].attrs['unit'] = 'nm'
        ds[str(names.bgroup)] = (bands, ['bands_vnir'] * len(BANDS))
        ds = ds.set_coords(str(names.bgroup))

        # Products state the reflectance unit as the CF dimensionless '1'.
        unit = ds_1km[BANDS[0]].attrs.get('units')
        if unit is not None:
            ds[self.payload_var].attrs['unit'] = str(unit)

        for source, dest in (
                ('Latitude', str(names.lat)),
                ('Longitude', str(names.lon)),
                ('View_Zenith', str(names.vza)),
                ('View_Azimuth', str(names.vaa))):
            ds[dest] = aux[source].astype('float32')
        ds['pixel_area_size'] = aux['Pixel_Area_Size'].astype('float32')
        area_unit = aux['Pixel_Area_Size'].attrs.get('units')
        if area_unit is not None:
            ds['pixel_area_size'].attrs['unit'] = str(area_unit)

        # 4. Solar angles, computed lazily from the geolocation ----------------
        dt = self._datetime(ds_1km)
        sza = da.map_blocks(_solar_zenith, ds[str(names.lat)].data,
                            ds[str(names.lon)].data, dt, dtype='float32')
        saa = da.map_blocks(_solar_azimuth, ds[str(names.lat)].data,
                            ds[str(names.lon)].data, dt, dtype='float32')
        ds[str(names.sza)] = ((rows, cols), sza)
        ds[str(names.saa)] = ((rows, cols), saa)

        # 5. Optional NWC/GEO cloud mask (2 km -> 1 km by 2x2 repetition) -----
        if self.cloudmask:
            ds['cloud'] = ((rows, cols),
                           self._cloud_mask(path, cy, cx))

        # 6. Canonical orientation: stored grid is north-up; flip once --------
        ds = ds.isel({rows: slice(None, None, -1)})
        ny, nx = ds.sizes[rows], ds.sizes[cols]
        ds = ds.assign_coords({rows: np.arange(ny), cols: np.arange(nx)})

        # 7. Curated attributes ------------------------------------------------
        filter_fn = (lambda x, y: x) if self.metadata_template is None \
            else filter_metadata
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): self._platform(path, ds_1km),
            str(names.product_name): path.stem,
            str(names.input_directory): str(path.parent),
            str(names.datetime): dt.isoformat(),
            str(names.resolution): 1000,
            str(names.shortname): path.stem.split('_')[0],
            str(names.crs): 'EPSG:4326',
            'auxfile': str(auxfile),
            'metadata': filter_fn(ds_1km.attrs, self.metadata_template),
        })
        return self.finalize(ds)

    # --------------------------------------------------------------- helpers

    def _find_aux(self, path: Path) -> Path:
        if self.auxfile is not None:
            if not self.auxfile.exists():
                raise ProductError(
                    f'GOESNG L1: auxiliary grid {self.auxfile} does not exist')
            return self.auxfile
        candidates = [path.parent / name for name in AUX_NAMES]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise ProductError(
            'GOESNG L1: no auxiliary grid found next to the product; looked for '
            + ', '.join(str(c) for c in candidates)
            + ' (pass auxfile= explicitly)')

    @staticmethod
    def _open_aux(auxfile: Path, cy: int, cx: int) -> xr.Dataset:
        if auxfile.suffix.lower() in ('.hdf', '.h4'):
            from eoread.hdf4 import load_hdf4
            aux = load_hdf4(auxfile, lazy=True, chunks=(cy, cx))
        else:
            # Explicit per-dim chunks (an int would chunk both axes alike,
            # breaking the y/x chunk alignment with the product) and the
            # h5netcdf engine: mixing in netCDF4 poisons later h5netcdf
            # dimension-scale writes in the same process (REFACTORING.md).
            aux = xr.open_dataset(
                auxfile, chunks={'Nlin': cy, 'Ncol': cx}, engine='h5netcdf')
        missing = [v for v in AUX_VARS if v not in aux]
        if missing:
            raise ProductError(
                f'GOESNG L1: auxiliary grid {auxfile.name} lacks {missing}')
        return aux

    @staticmethod
    def _datetime(ds_1km: xr.Dataset):
        start = parse(ds_1km.attrs['time_coverage_start'])
        stop = parse(ds_1km.attrs['time_coverage_end'])
        return start + (stop - start) / 2

    @staticmethod
    def _platform(path: Path, ds_1km: xr.Dataset) -> str:
        # The ``satellite`` scalar carries the platform id ('goes16').
        satellite = ds_1km['satellite'] if 'satellite' in ds_1km else None
        sat_id = str(satellite.attrs.get('id', '')) if satellite is not None else ''
        if not sat_id:
            match = re.search(r'_(goes\d+)_', path.name, re.IGNORECASE)
            sat_id = match.group(1).lower() if match else ''
        if sat_id.startswith('goes'):
            return 'GOES-' + sat_id[len('goes'):]
        return 'GOES-NG'

    def _cloud_mask(self, path: Path, cy: int, cx: int) -> da.Array:
        stem = path.stem
        match = re.search(r'_(goes\d+)_', path.name, re.IGNORECASE)
        satellite = match.group(1).upper() if match else 'GOES16'
        timestamp = stem.split('_')[-1]
        if len(timestamp) != 12:
            raise ProductError(
                f'GOESNG L1: cannot derive the acquisition time from {path.name!r}')
        # The CMA product name carries HHMMSS (the 10-minute slots are at
        # second 00); the Emultic name carries HHMM.
        cma_name = (f'S_NWC_CMA_{satellite}_globeE-NR_'
                    f'{timestamp[:8]}T{timestamp[8:]}00Z.nc')
        cma_path = path.with_name(cma_name)
        if not cma_path.exists():
            raise ProductError(
                f'GOESNG L1: cloudmask=True but the CMA file {cma_name} is '
                f'missing next to {path.name}')
        # 2 km chunks are half the target chunks, so the 2x2 repetition
        # lands on the same y/x chunk boundaries as the other fields.
        cma = xr.open_dataset(
            cma_path, chunks={'ny': max(1, cy // 2), 'nx': max(1, cx // 2)},
            engine='h5netcdf')['cma']
        # 2 km -> 1 km: repeat each pixel 2x2 (legacy semantics); NaN off disk
        # stays NaN.
        return da.repeat(da.repeat(cma.data, 2, axis=0), 2, axis=1).astype('float32')

    # ----------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        # `cloud` is optional (cloudmask=True only); FlagsInit keeps just the
        # variables actually present in the dataset.
        return [str(names.rtoa), str(names.vza), 'cloud']

    @classmethod
    def flag_dims_like(cls) -> str:
        return str(names.vza)

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.L1_INVALID:
            # Off-disk pixels are NaN in every band (fill value -32768).
            return ds[str(names.rtoa)].isnull().any(dim=str(names.bands))
        if flag_name == GenericFlags.CLOUD:
            if 'cloud' not in ds:
                raise ValueError(
                    'CLOUD flag not available: instantiate the reader with '
                    'cloudmask=True to attach the NWC/GEO cloud mask')
            return ds['cloud'] == 1
        raise ValueError(f'Unsupported GOESNG flag: {flag_name!r}')


def _solar_zenith(lat, lon, dt):
    """90 deg - solar altitude (pysolar); same definition as the legacy reader."""
    return 90. - pysol.get_altitude(lat, lon, dt, pressure=0.0, elevation=0.0)


def _solar_azimuth(lat, lon, dt):
    return pysol.get_azimuth(lat, lon, dt)
