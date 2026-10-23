#!/usr/bin/env python3
"""
PACE OCI (Ocean Color Instrument) reader, level L1B.

    >>> from eoread.readers.pace import Level1B_PACE_OCI       # doctest: +SKIP
    >>> ds = Level1B_PACE_OCI('PACE_OCI.20250430T...L1B.V3.nc')    # doctest: +SKIP

The product is a NetCDF-4 data tree with the ``geolocation_data``,
``sensor_band_parameters`` and ``observation_data`` groups. The payload
``Rtoa`` concatenates the blue / red / SWIR top-of-atmosphere reflectance
cubes; band labels are the integer-truncated central wavelengths,
deduplicated on first occurrence and then sorted by their string label — a
legacy quirk kept for parity (the old reader's comment says "increasing
wavelength" but the sort key is the label, so four-digit SWIR labels come
first). The dedup is also required by the canonical schema (band labels must
be unique): 291 raw bands down to 289.

Differences from the legacy v1 reader (documented deviations, see the
REFACTORING.md decision log):

- No orientation change: the product is natively south-up (row 0 at the
  southern edge of the swath), so v2 keeps the legacy layout as-is.
- Bands are chunked per band (v2 convention) instead of legacy's single
  all-bands block.
- The flag reader returns boolean masks: ``L1_INVALID`` is
  ``quality_flag != 0`` (legacy returned the raw float32 values; the flag
  pipeline applies ``condition != 0`` anyway, so the encoded flags are
  identical) and ``LAND`` is ``~water.astype(bool)`` — legacy's
  ``~ds['water']`` would raise a TypeError on the float32 watermask at
  compute time (latent bug fixed; identical result for the {0, 1} encoding).
- The ``shortname`` ('PACE_OCI') and ``crs`` ('EPSG:4326') attributes are now
  set (legacy did not).
- ``resolution`` keeps legacy's hardcoded 1200 m: the product carries no
  data-driven spacing field (measured along-track spacing on the sample is
  ~1.1 km; the cross-track spacing varies up to ~6 km near the poles).
- Azimuth angles (``saa`` / ``vaa``) keep the product's signed relative
  [-180, 180] convention; the naming registry declares [0, 180] — a documented
  registry inconsistency (see the OLCI/MSI/MERIS entries in REFACTORING.md).
"""

import re
from pathlib import Path

import xarray as xr
from dateutil.parser import parse

from core import log
from core.geo.naming import names
from core.tools import only

from eoread.flags import GenericFlags
from eoread.reader import Reader
from eoread.tools import filter_metadata


def _rename_bands(da: xr.DataArray) -> xr.DataArray:
    """Rename the band dimension to ``bands`` and add the band group coordinate.

    The coordinate values are the product's original band-dimension names
    (``blue_bands`` / ``red_bands`` / ``SWIR_bands``), as in legacy.
    """
    dim = only([d for d in da.dims if 'bands' in d])
    da = da.rename({dim: str(names.bands)})
    return da.assign_coords(
        {str(names.bgroup): (str(names.bands), [dim] * da.sizes[str(names.bands)])})


class Level1B_PACE_OCI(Reader):
    """PACE OCI level L1B: top-of-atmosphere reflectance, payload ``Rtoa``."""

    sensor = 'OCI'
    level = 'L1B'
    chunks_default = 500
    payload_var = str(names.rtoa)
    sample_products = ('PACE_L1B',)
    required_vars = (str(names.sza), str(names.saa), str(names.vza), str(names.vaa),
                     'water', 'quality_flag')
    flags = True
    VARIANTS = {'l1b': dict(detect=dict(field='processing_level', value='L1B'))}

    def read(self, path: Path) -> xr.Dataset:
        path = Path(path)
        assert path.exists(), f'{path} does not exist'

        if self.verbose:
            log.debug(f'Reading PACE OCI {self.level} product {path.name}')

        # Explicit h5netcdf engine, as in OLCI/ECOSTRESS: the process must stay
        # on a single libhdf5 (h5py's), since netCDF4's bundled copy loaded
        # first breaks h5py's dimension-scale writes downstream.
        tree = xr.open_datatree(path, engine='h5netcdf')
        # Loud on a non-L1B processing level.
        self.detect_variant(dict(tree.attrs))

        rows, cols, bands = str(names.rows), str(names.columns), str(names.bands)
        geo = tree['geolocation_data'].to_dataset().reset_coords(['latitude', 'longitude'])
        bdata = tree['sensor_band_parameters'].to_dataset()
        obs = tree['observation_data'].to_dataset()

        # Geolocation rasters, angles and masks (canonical names; the whole
        # group is cast to float32 as in legacy).
        ds = xr.Dataset().assign(geo.astype('float32'))
        ds = ds.rename_vars({
            'latitude': str(names.lat),
            'longitude': str(names.lon),
            'sensor_azimuth': str(names.vaa),
            'sensor_zenith': str(names.vza),
            'solar_azimuth': str(names.saa),
            'solar_zenith': str(names.sza),
            'watermask': 'water',
        })

        # TOA reflectance: the blue + red + SWIR cubes concatenated along bands.
        ds[str(names.rtoa)] = xr.concat(
            [_rename_bands(obs[k]) for k in ('rhot_blue', 'rhot_red', 'rhot_SWIR')],
            dim=bands)

        # Central wavelengths, same concatenation order.
        ds[str(names.cwav)] = xr.concat(
            [_rename_bands(bdata[k]) for k in
             ('blue_wavelength', 'red_wavelength', 'SWIR_wavelength')],
            dim=bands)

        # Band labels: integer-truncated wavelengths. Duplicates are dropped
        # (first occurrence kept) and the bands sorted by their string label —
        # a legacy quirk: four-digit SWIR labels come first (291 raw bands
        # down to 289; also required for unique band labels).
        ds = ds.assign_coords({bands: ds[str(names.cwav)].astype(int).astype(str)})
        labels = ds[bands].values
        seen = set()
        kept = [i for i, w in enumerate(labels) if w not in seen and not seen.add(w)]
        order = sorted(kept, key=lambda i: labels[i])
        ds = ds.isel({bands: order})

        # Canonical spatial dimensions; per-band chunks (v2 convention) plus
        # the requested spatial chunking.
        ds = ds.rename({'scans': rows, 'pixels': cols}).chunk(
            {bands: 1, rows: self.chunks[rows], cols: self.chunks[cols]})

        # Curated attributes (legacy parity + schema requirements); the
        # metadata copy is filtered when a template is given.
        time_start = parse(tree.attrs['time_coverage_start'])
        time_end = parse(tree.attrs['time_coverage_end'])
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): tree.attrs['platform'],
            str(names.product_name): path.name,
            str(names.input_directory): str(path.parent),
            # No data-driven spacing field exists in the product; legacy's
            # hardcoded value is kept (see module docstring).
            str(names.resolution): 1200,
            str(names.datetime): (time_start + (time_end - time_start) / 2).isoformat(),
            str(names.shortname): re.split(r'\d{8}', path.name)[0].rstrip('.'),
            str(names.crs): 'EPSG:4326',
            'metadata': dict(tree.attrs) if self.metadata_template is None else
                        filter_metadata(dict(tree.attrs), self.metadata_template),
        })

        return self.finalize(ds)

    # ----------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return ['quality_flag', 'water']

    @classmethod
    def flag_dims_like(cls) -> str:
        return 'quality_flag'

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """OCI L1B quality flags from the geolocation group.

        ``quality_flag`` carries the product's bit fields (Off_Earth /
        Solar_eclipse / Terrain_bad / 10_deg_scan_angle); any set bit marks an
        invalid pixel. The float32 ``water`` mask is 0 for land, 1 for water.
        """
        if flag_name == GenericFlags.L1_INVALID:
            # Canonical bool contract; legacy returned the raw float32 values,
            # which encode identically (raiseflag applies `condition != 0`).
            return ds['quality_flag'] != 0
        if flag_name == GenericFlags.LAND:
            # Legacy's `~ds['water']` raises TypeError on the float32 mask at
            # compute time; casting to bool first restores the intended
            # "not water" semantics for the {0, 1} encoding.
            return ~ds['water'].astype(bool)
        raise ValueError(f'Unsupported PACE flag: {flag_name!r}')
