#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPSO-1 level L1B reader: top-of-atmosphere radiance, payload ``Ltoa``.

HYPSO-1 (HYPerspectral Smallsat for Ocean Observation, NTNU) is a pushbroom
hyperspectral imager: each capture is a ``(lines, samples, bands)`` cube of
120 bands covering ~388-803 nm, with per-pixel geolocation and viewing/solar
angles on the same grid.

Two product layouts exist and are both supported:

- older products: ``navigation/`` group with ``latitude``, ``longitude``,
  ``sensor_zenith``, ``sensor_azimuth``, ``solar_zenith``, ``solar_azimuth``
  and the wavelengths as the ``wavelengths`` attribute of ``products/Lt``;
- current products (hypso-package writer): ``geometry/`` group with the same
  variable names; wavelengths may also be a ``metadata/corrections/wavelengths``
  variable or a ``products/wavelengths`` variable.

Deviations from the v1 implementation (removed in 2.0):

- Output is south-up (row 0 = southern edge, canonical orientation). The
  stored pushbroom grid is north-up (latitude decreases with ``lines``); all
  fields are flipped together once.
- ``Ltoa`` is float32 (v1 kept the file's uint16 counts). Integer
  radiance is preserved exactly by the cast.
- Band labels are the 0-based band indices as strings (v1 attempted
  ``bands.data.astype(str)`` on a coordinate that does not exist); the
  wavelengths are the ``cwav`` coordinate (from the file's own table).
- ``datetime`` is normalized to ISO from the file's ``date_aquired`` (v1
  kept the raw string; modern products use ``timestamp_acquired``).
- ``sensor``/``platform`` come from ``instrument_id``/``sat_id`` when the
  product provides them (modern files), falling back to the v1
  ``instrument`` / ``'HYPSO'`` values (the v1-compatible sample).
"""

from pathlib import Path

import numpy as np
import xarray as xr
from dateutil.parser import parse

from core import log
from core.geo.naming import names

from eoread.reader import ProductError, Reader
from eoread.tools import filter_metadata

#: Geolocation group names, in order of preference. Older L1B products use
#: ``navigation``; current hypso-package products use ``geometry``.
NAV_GROUPS = ('navigation', 'geometry')

#: Navigation variables required by the reader (same names in both groups).
NAV_VARS = {
    'latitude': str(names.lat),
    'longitude': str(names.lon),
    'sensor_zenith': str(names.vza),
    'sensor_azimuth': str(names.vaa),
    'solar_zenith': str(names.sza),
    'solar_azimuth': str(names.saa),
}


class Level1_HYPSO(Reader):
    """HYPSO-1 L1B hyperspectral radiance product."""

    sensor = 'HYPSO-1 Hyperspectral Imager'
    level = 'L1'
    chunks_default = 500
    payload_var = str(names.ltoa)
    sample_products = ('HYPSO_L1',)
    required_vars = (str(names.sza), str(names.saa), str(names.vza), str(names.vaa))
    flags = None

    # ------------------------------------------------------------------ read

    def read(self, path: Path) -> xr.Dataset:
        path = Path(path)
        if not path.exists():
            raise ProductError(f'HYPSO L1: {path} does not exist')
        if self.verbose:
            log.debug(f'Reading HYPSO L1 product {path.name}')

        # h5netcdf everywhere: mixing in netCDF4 poisons later h5netcdf
        # dimension-scale writes in the same process (REFACTORING.md).
        tree = xr.open_datatree(path, engine='h5netcdf')

        try:
            products = tree['products'].to_dataset()
        except KeyError as exc:
            raise ProductError(
                f'HYPSO L1: {path.name} carries no products group') from exc
        if 'Lt' not in products:
            raise ProductError(
                f'HYPSO L1: {path.name} carries no products/Lt radiance variable')

        lt = self._read_payload(products['Lt'])
        nav = self._read_navigation(tree, path.name)

        rows, cols, bands = str(names.rows), str(names.columns), str(names.bands)
        cy = self.chunks[rows]
        cx = self.chunks[cols]

        ds = xr.Dataset({self.payload_var: lt.chunk({bands: 1, rows: cy, cols: cx})})

        # Canonical spectral coordinates; the bands dim has no coordinate in the
        # file, so use the 0-based band indices as labels and the file's own
        # wavelength table (attribute or variable) as cwav.
        wavelengths = self._wavelengths(tree, products, path.name)
        nbands = ds.sizes[bands]
        if len(wavelengths) != nbands:
            raise ProductError(
                f'HYPSO L1: {len(wavelengths)} wavelengths for {nbands} bands '
                f'in {path.name}')
        ds = ds.assign_coords({bands: np.arange(nbands).astype(str)})
        ds[str(names.cwav)] = (bands, np.asarray(wavelengths, dtype='float64'))
        ds[str(names.cwav)].attrs['unit'] = 'nm'
        ds[str(names.bgroup)] = (bands, ['bands_vnir'] * nbands)
        ds = ds.set_coords(str(names.bgroup))

        unit = products['Lt'].attrs.get('units')
        if unit is not None:
            ds[self.payload_var].attrs['unit'] = str(unit)

        for source, dest in NAV_VARS.items():
            if source not in nav:
                raise ProductError(
                    f'HYPSO L1: {path.name} lacks the navigation field {source!r}')
            ds[dest] = nav[source].astype('float32').chunk({rows: cy, cols: cx})

        # Canonical orientation: the pushbroom grid is north-up; flip once.
        ds = ds.isel({rows: slice(None, None, -1)})
        ny, nx = ds.sizes[rows], ds.sizes[cols]
        ds = ds.assign_coords({rows: np.arange(ny), cols: np.arange(nx)})

        # Curated attributes ---------------------------------------------------
        attrs = tree.attrs
        filter_fn = (lambda x, y: x) if self.metadata_template is None \
            else filter_metadata
        ds.attrs.update({
            str(names.sensor): attrs.get('instrument_id', attrs.get(
                'instrument', 'HYPSO-1 Hyperspectral Imager')),
            str(names.platform): attrs.get('sat_id', 'HYPSO'),
            str(names.product_name): path.name,
            str(names.input_directory): str(path.parent),
            str(names.datetime): self._datetime(attrs),
            str(names.resolution): 40,
            str(names.shortname): attrs.get('sat_id', 'HYPSO-1'),
            str(names.crs): 'EPSG:4326',
            'metadata': filter_fn(attrs, self.metadata_template),
        })
        return self.finalize(ds)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _read_payload(lt: xr.DataArray) -> xr.DataArray:
        """Return the radiance cube as (bands, y, x) float32."""
        dims = lt.dims
        required = {'lines', 'samples', 'bands'}
        if set(dims) != required:
            raise ProductError(
                f'HYPSO L1: unexpected products/Lt dims {dims}; expected '
                f'{sorted(required)}')
        # The file stores integer radiance counts; the cast is exact.
        return (lt.astype('float32')
                .transpose('bands', 'lines', 'samples')
                .rename({'bands': str(names.bands), 'lines': str(names.rows),
                         'samples': str(names.columns)}))

    @staticmethod
    def _read_navigation(tree: xr.DataTree, name: str) -> xr.Dataset:
        for group in NAV_GROUPS:
            if group in tree.children:
                nav = tree[group].to_dataset()
                break
        else:
            raise ProductError(
                f'HYPSO L1: {name} carries neither a navigation nor a geometry '
                f'group (looked for {NAV_GROUPS})')
        missing = [v for v in NAV_VARS if v not in nav]
        if missing:
            raise ProductError(
                f'HYPSO L1: {name} navigation group lacks {missing}')
        return nav.rename({'lines': str(names.rows),
                           'samples': str(names.columns)})

    @staticmethod
    def _wavelengths(tree: xr.DataTree, products: xr.Dataset, name: str) -> np.ndarray:
        # 1. Oldest layout: attribute on products/Lt.
        attr = products['Lt'].attrs.get('wavelengths')
        if attr is not None:
            return np.asarray(attr, dtype='float64')
        # 2. A products/wavelengths variable.
        if 'wavelengths' in products:
            return np.asarray(products['wavelengths'].values, dtype='float64')
        # 3. Current hypso-package layout: metadata/corrections/wavelengths.
        corrections = tree.children.get('metadata', None)
        if corrections is not None and 'corrections' in corrections.children:
            corr = corrections['corrections'].to_dataset()
            if 'wavelengths' in corr:
                return np.asarray(corr['wavelengths'].values, dtype='float64')
        raise ProductError(
            f'HYPSO L1: {name} carries no wavelength table (Lt.wavelengths '
            f'attribute, products/wavelengths or metadata/corrections/wavelengths)')

    @staticmethod
    def _datetime(attrs: dict) -> str:
        # Legacy key has the historical typo; modern products use the full name.
        raw = attrs.get('timestamp_acquired', attrs.get('date_aquired'))
        if raw is None:
            raise ProductError(
                'HYPSO L1: no acquisition time (timestamp_acquired/date_aquired)')
        return parse(str(raw)).isoformat()
