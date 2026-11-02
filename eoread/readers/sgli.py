#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCOM-C SGLI level L1B reader: top-of-atmosphere reflectance, payload ``Rtoa``.

The product (e.g. ``GC1SG1_201912050159F05712_1BSG_VNRDQ_1007.h5``) stores
per-band top-of-atmosphere radiances (``Image_data/Lt_VNxx``, uint16 counts),
a land/water percentage mask and a quality flag on the full grid, plus
geodetic lat/lon and viewing/solar angles on a tie-point grid
(``Resampling_interval = 10`` pixels) with per-field ``Slope`` / ``Offset``
scale factors.

v2 applies the full affine transform ``Slope * x + Offset`` to the tie fields
— legacy applied ``(1 + Slope) * x``, which doubled lat/lon (Slope = 1,
Offset = 0) and inflated every centidegree-encoded angle ~101x (Slope = 0.01,
Offset = 0).

Deviations from legacy (see REFACTORING.md):

- The output is south-up: row 0 is the southern edge of the scene (canonical
  orientation). The file stores lines in acquisition order, so granules on a
  descending ground track come out north-up and are flipped; ascending
  granules are already south-up.
- No dtype is ever coerced: each variable keeps whatever type its read and
  arithmetic naturally produce — float32 for the tie-point fields (file-
  stored float32/int16 with float32 scale factors) and float64 for the
  full-grid geometry and `Rtoa` (float64 interpolation weights / cos(sza)),
  as in legacy.
- ``shortname`` / ``crs`` attributes added; ``datetime`` normalized to ISO
  from the product's non-ISO ``Scene_center_time`` attribute (same instant).
- Band labels keep the product's stored-channel names with the parentheses of
  the sub-band code turned into an underscore (``VN01(S06)`` → ``VN01_S06``);
  the canonical schema rejects other punctuation in band labels. The raw names
  remain in the ``metadata`` attribute, and the stored order is kept — it is
  already ascending central wavelength.
- The legacy ``add_ancillary_data`` option is dropped: the Ancillary_data
  group is instrument telemetry with no consumer in eoread.
"""

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dateutil.parser import parse

from core import log
from core.geo.naming import names

from eoread.flags import GenericFlags
from eoread.reader import Reader
from eoread.tools import filter_metadata, spatial_resample

#: Central wavelengths (nm) of VN01..VN11 in band order. Legacy carried the
#: same values as a hardcoded table; its SRF-file-based computation was dead
#: code (referenced an undefined name) and is not ported.
SGLI_CENTRAL_WAVELENGTHS = (380., 412., 443., 490., 530., 565.,
                            673.5, 673.5, 763., 868.5, 868.5)

#: Geometry_data tie-point fields -> output names (legacy's exact names).
TIE_VARS = {
    'Latitude': 'lat_tie',
    'Longitude': 'lon_tie',
    'Sensor_zenith': str(names.vza) + '_tie',
    'Sensor_azimuth': str(names.vaa) + '_tie',
    'Solar_zenith': str(names.sza) + '_tie',
    'Solar_azimuth': str(names.saa) + '_tie',
}

#: Tie field -> full-grid canonical name.
TIE_TO_FULL = {
    'lat_tie': str(names.lat),
    'lon_tie': str(names.lon),
    str(names.vza) + '_tie': str(names.vza),
    str(names.vaa) + '_tie': str(names.vaa),
    str(names.sza) + '_tie': str(names.sza),
    str(names.saa) + '_tie': str(names.saa),
}


class Level1B_SGLI(Reader):
    """GCOM-C SGLI level L1B: top-of-atmosphere reflectance, payload ``Rtoa``."""

    sensor = 'SGLI'
    level = 'L1B'
    chunks_default = 500
    payload_var = str(names.rtoa)
    sample_products = ('SGLI_L1B',)
    required_vars = (str(names.sza), str(names.saa), str(names.vza), str(names.vaa),
                     'water', 'quality_flag')
    flags = True
    VARIANTS = {'l1b': dict(detect=dict(field='Product_level', value='Level-1B'))}

    def read(self, path: Path) -> xr.Dataset:
        path = Path(path)
        assert path.exists(), f'{path} does not exist'

        if self.verbose:
            log.debug(f'Reading SGLI {self.level} product {path.name}')

        # Explicit h5netcdf engine, as in OLCI/ECOSTRESS/PACE: the process must
        # stay on a single libhdf5 (h5py's), since netCDF4's bundled copy loaded
        # first breaks h5py's dimension-scale writes downstream. phony_dims:
        # some groups carry variables without dimension scales.
        tree = xr.open_datatree(path, engine='h5netcdf', phony_dims='sort')
        metadata = dict(tree['Global_attributes'].attrs)
        # Loud on a non-L1B product level.
        self.detect_variant(metadata)

        rows, cols, bands = str(names.rows), str(names.columns), str(names.bands)

        # Image group: per-band radiances + masks. Canonical spatial dims via
        # positional rename; the file's dim names are not part of the contract.
        imdata = tree['Image_data'].to_dataset()
        imdata = imdata.rename_dims(dict(zip(imdata['Lt_VN01'].dims, (rows, cols))))
        shape = dict(imdata['Lt_VN01'].sizes)

        # Tie-point geometry group. The group also carries ~56 per-band
        # telemetry fields we do not emit; keep only the tie fields and load
        # them eagerly (small, and the output keeps them unchunked). Loading
        # before the orientation flip below is required: a negative-step slice
        # on the lazy h5netcdf arrays silently returns an empty selection
        # instead of raising.
        geom = tree['Geometry_data'].to_dataset()[list(TIE_VARS)]
        geom = geom.rename_dims(
            dict(zip(geom['Latitude'].dims, (rows + '_tie', cols + '_tie'))))
        geom = geom.load()

        # Canonical orientation: row 0 is the southern edge. The file stores
        # lines in acquisition order, so a descending ground track comes out
        # north-up; detect it from the tie latitudes and flip only when needed
        # (ascending granules are already south-up).
        lat_raw = geom['Latitude'].values
        north_up = float(lat_raw[0].mean()) > float(lat_raw[-1].mean())
        if north_up:
            geom = geom.isel({rows + '_tie': slice(None, None, -1)})

        ds = xr.Dataset()
        # Band labels: the product's stored-channel names carry a parenthesized
        # sub-band code ('VN01(S06)'); the canonical schema requires scientific
        # names (letters, digits, underscore), so the parens become an
        # underscore. The raw names stay in the metadata attribute.
        channels = [c.replace('(', '_').replace(')', '')
                    for c in metadata['Stored_channels'].split(',')]
        ds = ds.assign_coords({bands: channels})
        ds = ds.assign({str(names.cwav): ((bands,),
                                          np.array(SGLI_CENTRAL_WAVELENGTHS,
                                                   dtype='float32'))})

        # Tie-point fields: full affine transform (see module docstring), then
        # lazy interpolation onto the full grid.
        for src, dst in TIE_VARS.items():
            ds[dst] = geom[src]
        delta = 10
        for dst in TIE_VARS.values():
            assert int(ds[dst].attrs['Resampling_interval']) == delta, \
                f'{dst}: unexpected tie-point spacing'
            # No cast: the file stores tie data as float32/int16 with
            # float32 attrs, so the affine transform stays in float32.
            ds[dst] = (ds[dst].attrs['Slope'] * ds[dst] + ds[dst].attrs['Offset'])

        # Tie coordinates in pixel units (tie line i sits at row i*delta); the
        # interpolation targets are the full-grid pixel positions.
        ds[cols + '_tie'] = np.arange(ds.sizes[cols + '_tie']) * delta
        ds[rows + '_tie'] = np.arange(ds.sizes[rows + '_tie']) * delta

        # Chunk and orient the bands first so every full-grid variable below
        # can share one spatial chunking (the dataset contract requires
        # consistent chunks along shared dims). Chunk before the flip: a
        # negative-step slice on the lazy h5netcdf arrays silently returns
        # empty selections instead of raising (same trap as the tie fields
        # above). The dask-level row reversal keeps each block on the file's
        # row lattice (it reverses the block order); do NOT re-chunk to the
        # unflipped grid afterwards — with ny % chunk != 0 it is offset from
        # the file lattice and every block would straddle two file blocks
        # (2x the HDF5 reads).
        imdata = imdata.chunk(self.chunks)
        if north_up:
            imdata = imdata.isel({rows: slice(None, None, -1)})
        first_band = next(iter(imdata.data_vars))
        spatial_chunks = {rows: imdata[first_band].chunks[0],
                          cols: imdata[first_band].chunks[1]}

        for tie_name, name in TIE_TO_FULL.items():
            a = ds[tie_name].rename({rows + '_tie': rows, cols + '_tie': cols})
            # The interpolation branch returns the historical swapped layout;
            # transpose back to (rows, columns). The linear weights are
            # float64, so cast back to the tie data's float32 — legacy's
            # spatial_resample did the same (its output was float32). Re-chunk
            # onto the bands' file-aligned grid (in-memory re-partition only):
            # a dask elementwise op on mismatched chunks refines to the union
            # grid, which would pull Lt off its file-aligned blocks in the
            # reflectance division below.
            ds[name] = spatial_resample(a, shape, chunks=self.chunks) \
                              .transpose(rows, cols).astype('float32') \
                              .chunk(spatial_chunks)

        # TOA reflectance: per-band counts -> reflectance, stacked on bands
        # (per-band chunks, as in legacy's merge output). mus shares the
        # bands' chunking by construction (angles re-chunked above), so the
        # division stays block-aligned.
        mus = da.cos(da.radians(ds[str(names.sza)]))

        parts = []
        first_attrs = None
        for band in ds[bands].values:
            lt = imdata[f'Lt_{band[:4]}']
            if first_attrs is None:
                first_attrs = dict(lt.attrs)
            # No cast: counts * float32 attrs stay float32, and mus is
            # float32 (angles cast above), so reflectance lands in float32,
            # as in legacy.
            part = (((lt & np.uint16(lt.attrs['Mask']))
                     * lt.attrs['Slope_reflectance']
                     + lt.attrs['Offset_reflectance']) / mus)
            part.attrs = {}
            parts.append(part)
        # No re-chunk after the concat: the parts already carry the bands'
        # file-aligned chunks (one per band), and chunking back to the
        # unflipped grid would offset every block from the file lattice.
        ds[str(names.rtoa)] = xr.concat(parts, dim=bands)
        # Legacy's merge kept the first band's attributes (including its raw
        # lowercase-free `Unit`); reflectance is dimensionless and gets no
        # `unit` attribute of its own.
        ds[str(names.rtoa)].attrs.update(first_attrs)

        # Masks (same full grid).
        ds['water'] = imdata['Land_water_flag']
        ds['quality_flag'] = imdata['QA_flag']

        ds = ds.assign_coords(
            {str(names.bgroup): (bands, ['bands'] * ds.sizes[bands])})
        ds = ds.transpose(..., rows, cols)

        # Curated attributes (legacy parity + schema requirements); the
        # metadata copy is filtered when a template is given.
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): 'GCOM-C',
            str(names.product_name): path.name,
            str(names.input_directory): str(path.parent),
            # No data-driven spacing field exists in the product; legacy's
            # hardcoded value is kept.
            str(names.resolution): 250,
            # The file stores a non-ISO 'YYYYMMDD HH:MM:SS.mmm'; same instant.
            str(names.datetime): parse(metadata['Scene_center_time']).isoformat(),
            str(names.shortname): path.name.split('_')[2],
            str(names.crs): 'EPSG:4326',
            # SRF getter attributes carried over from legacy (consumed by eotools).
            '_srf_getter': 'eotools.srf.get_SRF_eumetsat',
            '_srf_getter_arg': 'gcom-c_1_sgli',
            'metadata': metadata if self.metadata_template is None else
                        filter_metadata(metadata, self.metadata_template),
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
        """GCOM-C SGLI L1B flags.

        ``quality_flag`` (QA_flag, uint16) bit 0 is the per-pixel
        channel-integrity bit — the product's own description says
        ``1 : Integrity`` (sound), and the sample scene has it set on ~86% of
        pixels — so L1_INVALID is the *negation* of bit 0; legacy's raw-value
        test flagged the sound majority instead. ``water`` (Land_water_flag,
        uint8) is a land percentage 0-100 (0 = water, 100 = land); legacy's
        ``~ds['water']`` on the uint8 mask is nonzero everywhere, so its LAND
        flag was set on every pixel.
        """
        if flag_name == GenericFlags.L1_INVALID:
            # Bit 0 set = channel integrity (sound), per the product's own bit
            # description — invalid is the negation of bit 0. Legacy's raw-value
            # test flagged the sound majority instead (pinned by
            # test_l1b_flag_values).
            return (ds['quality_flag'] & 1) == 0
        if flag_name == GenericFlags.LAND:
            # Majority-land threshold on the land percentage.
            return ds['water'] >= 50
        raise ValueError(f'Unsupported SGLI flag: {flag_name!r}')
