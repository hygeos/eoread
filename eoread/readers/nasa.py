#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NASA L1C (OCSSW l2gen) reader.

Reads Level 1C products generated with SeaDAS/OCSSW `l2gen` for any NASA ocean
color sensor (MODIS, VIIRS, SeaWiFS, Sentinel-3 OLCI, ...): a single NetCDF4
file with `/sensor_band_parameters`, `/navigation_data` and `/geophysical_data`
groups. The payload is the per-band TOA reflectance `Rtoa_b = rhot_b /
polcor_b`, masked where `rhot_b <= -100` (v1 mask, verbatim).

v2 port of the v1 `eoread.nasa.Level1_NASA` reader (REFACTORING.md §9/§10).
Deviations from v1:

- Per-file data-driven orientation check (first/last *valid* latitude along
  the center column, longitude along the center row) instead of a per-sensor
  reversal table; the legacy table was never validated against the v2
  canonical (south-up) convention and unknown instruments just work.
- Per-file data-driven resolution estimate (median adjacent-pixel ground
  spacing along a center row) instead of a hard-coded per-sensor value.
- The band table is sliced (`wavelength[:number_of_reflective_bands]`); the
  legacy code indexed it with a scalar, so it could read at most one band.
- `time_coverage_start/end` are parsed with `datetime.fromisoformat` (robust
  to missing fractional seconds) and `datetime` is the interval midpoint,
  tz-aware ISO.
- Loud errors on missing required variables/attributes instead of silent
  skips; no mus/muv/raa geometry (v2 convention); flags are decoded on demand
  (`flags=True`, `_flag_reader` stamped by `finalize`).

The engine is pinned to netCDF4: the product is plain NetCDF, and opening it
through h5netcdf/h5py in a process that also writes HDF5 products would mix
HDF5 library copies.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from hygeos_core.geo.naming import names

from eoread.flags import GenericFlags
from eoread.reader import ProductError, Reader
from eoread.tools import filter_metadata


class Level1_NASA(Reader):
    """NASA L1C OCSSW l2gen product (any sensor).

    The payload is the per-band TOA reflectance `Rtoa_b = rhot_b / polcor_b`,
    masked where `rhot_b <= -100`. Bands are labeled B1..Bn in ascending
    central wavelength (`cwav`). Canonical orientation (south-up /
    west-to-east) and the `resolution` attribute are estimated per file from
    the navigation grid — no per-sensor table is used.

    Quality flags decode through `Level1_NASA.getflag` from the raw
    `l2_flags` bit field (its `flag_meanings`/`flag_masks` attributes carry
    the per-product bit definitions).
    """

    sensor = 'NASA'
    level = 'L1'
    chunks_default = 500
    payload_var = str(names.rtoa)
    required_vars = (str(names.rtoa), 'l2_flags')
    flags = True

    # ------------------------------------------------------------------- read

    def read(self, filepath: Path) -> xr.Dataset:
        rows, cols = str(names.rows), str(names.columns)
        bands = str(names.bands)
        filepath = Path(filepath)

        # 0. Root attributes + band table (closed once extracted) --------------
        root = xr.open_dataset(filepath, chunks={}, engine='netcdf4')
        attrs = dict(root.attrs)
        sbp = xr.open_dataset(
            filepath, group='/sensor_band_parameters', chunks={},
            engine='netcdf4')
        # number_of_reflective_bands is a dimension in the l2gen L1C files (a
        # scalar variable in some other versions)
        if 'number_of_reflective_bands' in sbp.sizes:
            nref = sbp.sizes['number_of_reflective_bands']
        else:
            nref = int(sbp['number_of_reflective_bands'].values)
        wav_table = np.asarray(sbp['wavelength'].values, dtype='f8')
        root.close()
        sbp.close()

        # 1. Instrument / platform (loud) --------------------------------------
        instrument = str(attrs.get('instrument') or '').strip()
        platform = str(attrs.get('platform') or '').strip()
        if not instrument:
            raise ProductError(
                f'{self.sensor} {self.level}: missing/empty "instrument" '
                f'attribute in {filepath.name}')
        if not platform:
            raise ProductError(
                f'{self.sensor} {self.level}: missing/empty "platform" '
                f'attribute in {filepath.name}')

        # 2. Datetime: midpoint of the coverage interval (tz-aware) -------------
        try:
            dstart = datetime.fromisoformat(
                str(attrs['time_coverage_start']).replace('Z', '+00:00'))
            dstop = datetime.fromisoformat(
                str(attrs['time_coverage_end']).replace('Z', '+00:00'))
        except (KeyError, ValueError) as e:
            raise ProductError(
                f'{self.sensor} {self.level}: missing or unparseable '
                f'time_coverage_start/end attributes in {filepath.name}') from e
        dt_iso = (dstart + (dstop - dstart) // 2).isoformat()

        # 3. Band order: ascending cwav, labels B1..Bn ---------------------------
        if nref <= 0 or wav_table.size < nref:
            raise ProductError(
                f'{self.sensor} {self.level}: the band table has '
                f'{wav_table.size} wavelengths but number_of_reflective_bands '
                f'is {nref}')
        wav = wav_table[:nref]
        order = np.argsort(wav, kind='stable')
        # The per-band variables are named after the int-truncated wavelength
        # (v1 parity: astype('int32')).
        suffixes = [int(w) for w in wav]

        # 4. Navigation + geophysical groups (chunked, kept open) ---------------
        chunkmap = {'number_of_lines': self.chunks[rows],
                    'pixels_per_line': self.chunks[cols]}
        geo = xr.open_dataset(
            filepath, group='/geophysical_data', chunks=chunkmap, engine='netcdf4')
        navi = xr.open_dataset(
            filepath, group='/navigation_data', chunks=chunkmap, engine='netcdf4')

        # Some OCSSW versions store navigation on control points; only the
        # one-control-point-per-pixel case is supported (v1 parity).
        if 'pixel_control_points' in navi.dims:
            nx_geo = geo.sizes['pixels_per_line']
            ny_geo = geo.sizes['number_of_lines']
            cols_ok = 'cntl_pt_cols' in navi.variables and bool(
                np.array_equal(navi.cntl_pt_cols.values,
                               np.arange(1, nx_geo + 1)))
            rows_ok = 'cntl_pt_rows' in navi.variables and bool(
                np.array_equal(navi.cntl_pt_rows.values,
                               np.arange(1, ny_geo + 1)))
            if not (navi.sizes['pixel_control_points'] == nx_geo
                    and cols_ok and rows_ok):
                raise ProductError(
                    f'{self.sensor} {self.level}: navigation stored at '
                    f'subsampled control points is not supported (one control '
                    f'point per pixel required)')
            navi = navi.drop_vars(['cntl_pt_cols', 'cntl_pt_rows'],
                                  errors='ignore') \
                .rename_dims({'pixel_control_points': 'pixels_per_line'})

        geo = geo.rename_dims(
            {'number_of_lines': rows, 'pixels_per_line': cols})
        navi = navi.rename_dims(
            {'number_of_lines': rows, 'pixels_per_line': cols})

        # 5. Payload: Rtoa_b = rhot/polcor, masked where rhot <= -100 -----------
        parts = []
        for pos in order:
            rvar, pvar = f'rhot_{suffixes[pos]}', f'polcor_{suffixes[pos]}'
            if rvar not in geo.variables or pvar not in geo.variables:
                raise ProductError(
                    f'{self.sensor} {self.level}: missing band variable '
                    f'{rvar!r}/{pvar!r} in /geophysical_data')
            rhot, polcor = geo[rvar], geo[pvar]
            parts.append((rhot / polcor).where(rhot > -100.))
        ds = xr.Dataset()
        ds[str(names.rtoa)] = xr.concat(parts, dim=bands)

        # 6. Geolocation + angles + flags (loud on missing) ----------------------
        for var in ('latitude', 'longitude'):
            if var not in navi.variables:
                raise ProductError(
                    f'{self.sensor} {self.level}: missing {var!r} in '
                    f'/navigation_data')
        for param in ('solz', 'senz', 'sola', 'sena'):
            if param not in geo.variables:
                raise ProductError(
                    f'{self.sensor} {self.level}: missing angle variable '
                    f'{param!r} in /geophysical_data')
        if 'l2_flags' not in geo.variables:
            raise ProductError(
                f'{self.sensor} {self.level}: missing l2_flags in '
                f'/geophysical_data')

        ds[str(names.lat)] = navi['latitude']
        ds[str(names.lon)] = navi['longitude']
        ds[str(names.sza)] = geo['solz']
        ds[str(names.vza)] = geo['senz']
        ds[str(names.saa)] = geo['sola']
        ds[str(names.vaa)] = geo['sena']
        ds['l2_flags'] = geo['l2_flags']

        # 7. Band coordinates ------------------------------------------------------
        cwav = wav[order]
        ds = ds.assign_coords({
            bands: [f'B{i + 1}' for i in range(nref)],
            str(names.cwav): (bands, cwav),
            str(names.bgroup): (bands, ['bands_vnir'] * nref),
        })
        ds[str(names.cwav)].attrs['unit'] = 'nm'

        # 8. Orientation + chunk consistency -----------------------------------------
        ds = self._orient(ds)
        ds = ds.unify_chunks()

        # 9. Attributes ------------------------------------------------------------------
        short = attrs.get('short_name')
        if not isinstance(short, str) or not short.strip():
            short = f'L1C_{instrument.upper()}'
        ds.attrs = {}
        filter_fn = (lambda x, y: x) if self.metadata_template is None \
            else filter_metadata
        ds.attrs.update({
            str(names.sensor): instrument,
            str(names.platform): platform,
            str(names.product_name): filepath.stem,
            str(names.datetime): dt_iso,
            str(names.input_directory): str(filepath.parent),
            str(names.resolution): self._resolution(ds),
            str(names.shortname): short,
            str(names.crs): 'EPSG:4326',
            '_srf_getter': 'eotools.srf.get_SRF_NASA',
            '_srf_getter_arg': f'{platform}_{instrument}'.lower(),
        })
        ds.attrs['metadata'] = filter_fn(attrs, self.metadata_template)

        return self.finalize(ds)

    # ------------------------------------------------------------- orientation

    def _orient(self, ds: xr.Dataset) -> xr.Dataset:
        """Flip to the canonical south-up / west-to-east orientation, per file.

        The OCSSW geolocation is NaN outside the valid swath (edge rows may be
        fully or partially invalid), so each flip is decided from the
        first/last *valid* latitude along the center column (longitude along
        the center row). Each check reads one thin line of the grid.
        """
        rows, cols = str(names.rows), str(names.columns)
        ny, nx = ds.sizes[rows], ds.sizes[cols]

        def endpoints(var: xr.DataArray, dim: str, pos: int) -> tuple[float, float]:
            vals = var.isel({dim: pos}).values
            valid = np.flatnonzero(np.isfinite(vals))
            if valid.size < 2:
                raise ProductError(
                    f'{self.sensor} {self.level}: cannot determine orientation '
                    f'({var.name} has fewer than 2 valid values along {dim})')
            return float(vals[valid[0]]), float(vals[valid[-1]])

        lat_first, lat_last = endpoints(ds[str(names.lat)], cols, nx // 2)
        if lat_first > lat_last:
            ds = ds.isel({rows: slice(None, None, -1)})
        lon_first, lon_last = endpoints(ds[str(names.lon)], rows, ny // 2)
        if lon_first > lon_last:
            ds = ds.isel({cols: slice(None, None, -1)})
        ny, nx = ds.sizes[rows], ds.sizes[cols]
        return ds.assign_coords({rows: np.arange(ny), cols: np.arange(nx)})

    # -------------------------------------------------------------- resolution

    def _resolution(self, ds: xr.Dataset) -> int:
        """Data-driven ground sample distance (m).

        Median adjacent-pixel ground spacing along a center row of the
        geolocation grid (consecutive valid pixels only); a small candidate
        loop around the grid center guards against a fully invalid row.
        """
        rows, cols = str(names.rows), str(names.columns)
        ny, nx = ds.sizes[rows], ds.sizes[cols]
        lat = ds[str(names.lat)]
        lon = ds[str(names.lon)]
        for i in (ny // 2, ny // 2 - 1, ny // 2 + 1, ny // 2 - 2, ny // 2 + 2):
            if not 0 <= i < ny:
                continue
            la = lat.isel({rows: i}).values
            lo = lon.isel({rows: i}).values
            if not np.any(np.isfinite(la)):
                continue
            dlat = np.diff(la) * 111320.
            midlat = float(np.nanmean(la))
            dlon = np.diff(lo) * 111320. * np.cos(np.deg2rad(midlat))
            d = np.hypot(dlat, dlon)
            d = d[np.isfinite(d)]
            if d.size:
                return int(round(float(np.median(d))))
        raise ProductError(
            f'{self.sensor} {self.level}: cannot estimate the resolution '
            f'from the geolocation')

    # ----------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return ['l2_flags']

    @classmethod
    def flag_dims_like(cls) -> str:
        return 'l2_flags'

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """Decode `flag_name` from the raw `l2_flags` bit field.

        The bit definitions come from the variable's own
        `flag_meanings`/`flag_masks` attributes (per-product in OCSSW
        products): LAND, and L1_INVALID = NAVFAIL + BOWTIEDEL.
        """
        mapping = {
            GenericFlags.LAND: ('LAND',),
            GenericFlags.L1_INVALID: ('NAVFAIL', 'BOWTIEDEL'),
        }
        if flag_name not in mapping:
            raise ValueError(f'Unsupported NASA L1C flag: {flag_name!r}')
        flags = ds['l2_flags']
        meanings = flags.attrs.get('flag_meanings')
        masks = flags.attrs.get('flag_masks')
        # (flag_masks is read as a numpy array from the netcdf files)
        if not isinstance(meanings, str) or not isinstance(masks, (str, list, tuple, np.ndarray)):
            raise ProductError(
                f'{cls.sensor} {cls.level}: l2_flags carries no '
                f'flag_meanings/flag_masks attributes')
        meanings = meanings.replace(',', ' ').split()
        if isinstance(masks, str):
            masks = masks.replace(',', ' ').split()
        else:
            masks = [str(m) for m in masks]
        table = dict(zip(meanings, masks))
        missing = [f for f in mapping[flag_name] if f not in table]
        if missing:
            raise ProductError(
                f'{cls.sensor} {cls.level}: l2_flags has no definition for '
                f'bit(s) {missing}')
        mask = sum(int(table[f]) for f in mapping[flag_name])
        return (flags & mask) != 0
