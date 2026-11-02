#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODIS L1B swath reader (Phase 4).

Reads the HDF4 native scene (MOD021KM / MYD021KM) either through the netCDF4
stack when it was built with HDF4 support, or -- loudly -- through the pyhdf
fallback (`eoread.hdf4.load_hdf4`). The legacy reader
(`eoread/legacy/modis.py`) is kept as the reference implementation.

All four radiance blocks of a MYD021KM/MOD021KM scene share the single ~1 km
registration ('Aggr1km'), so the payload is one (bands, y, x) grid; the coarse
geolocation grid is kept as (y_tie, x_tie) for the raw tie-point angles, and
lat/lon/gflags are upsampled to the pixel grid.

Documented breaks from v1 (see REFACTORING.md):
- radiance fixed to `scale * (raw - offset)`; v1 used `scale * raw + offset`,
  which yields unphysical values;
- BT, the Uncert_Indexes/Samples_Used blocks and the redundant EV_Band26 /
  Height / Range datasets are dropped; the contract keeps Ltoa + Rtoa;
- the `resolution` reader parameter is dropped: v1 never resampled (the
  parameter was a misnomer); `resolution` is stamped as a 1000 m nominal
  attribute instead;
- datetime is the acquisition start parsed from the granule filename; the file
  only carries a time-of-day range and the production datetime, which v1
  stored as the product datetime;
- band labels are scientific names: MODIS band numbers as ``B1``...``B36``,
  with the high-gain TDI pairs '13.5'/'14.5' as ``B13HG``/``B14HG`` (the
  file-native values remain the internal wavelength-table lookup keys);
  bands are sorted by ascending cwav (file block order is not wavelength-
  ordered); cwav comes from the ATBD v7 measured central wavelengths (nm),
  aligned by construction.
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from core import log
from core.geo.naming import names

from eoread.flags import GenericFlags
from eoread.hdf4 import load_hdf4
from eoread.reader import ProductError, Reader, UnsupportedProductType
from eoread.tools import filter_metadata, parse_hdf4_metadata


#: Measured central wavelengths (nm), ATBD v7 Table 1 (Aqua column). The
#: high-gain TDI bands 13.5/14.5 share the measured center of their low-gain
#: partners 13/14 (same detector channel, different gain setting).
_CWVL = {
    '1.0': 645.8, '2.0': 856.9, '3.0': 466.1, '4.0': 553.9,
    '5.0': 1241.5, '6.0': 1628.1, '7.0': 2113.9,
    '8.0': 412.4, '9.0': 442.2, '10.0': 487.4, '11.0': 530.1, '12.0': 547.2,
    '13.0': 666.0, '13.5': 666.0, '14.0': 677.6, '14.5': 677.6,
    '15.0': 746.8, '16.0': 866.9, '17.0': 904.4, '18.0': 936.4, '19.0': 936.3,
    '20.0': 3780.2, '21.0': 3981.8, '22.0': 3972.0, '23.0': 4061.6,
    '24.0': 4448.3, '25.0': 4526.3, '26.0': 1382.3,
    '27.0': 6786.8, '28.0': 7349.3, '29.0': 8555.3, '30.0': 9723.7,
    '31.0': 11026.2, '32.0': 12042.3, '33.0': 13364.7, '34.0': 13685.9,
    '35.0': 13925.2, '36.0': 14215.2,
}

#: Product user guide (M1054E), same reference as the legacy reader.
_USER_GUIDE = ('https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/'
               'M1054E_PUG_2022_1005_V6.2.2_Terra_V6.2.3_Aqua.pdf')

#: Radiance blocks in file (concat) order. The emissive block carries no
#: reflectance calibration, so its Rtoa is NaN by construction.
_EVBLOCKS = ('EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB',
             'EV_1KM_RefSB', 'EV_1KM_Emissive')

#: File dim names (the ':MODIS_SWATH_Type_L1B' scale suffix is stripped first)
#: to canonical names. The four band dims keep the file block names, reused as
#: the values of the `bands_group` coordinate.
_DIM_MAP = {
    '10*nscans': str(names.rows),
    'Max_EV_frames': str(names.columns),
    '2*nscans': str(names.rows) + '_tie',
    '1KM_geo_dim': str(names.columns) + '_tie',
    'Band_250M': 'bands_250M',
    'Band_500M': 'bands_500M',
    'Band_1KM_RefSB': 'bands_1KM_RefSB',
    'Band_1KM_Emissive': 'bands_1KM_Emissive',
}

#: File variable names to canonical names (the raw tie-point angles keep the
#: coarse grid; no full-grid angles are emitted).
_VAR_MAP = {
    'Latitude': str(names.lat),
    'Longitude': str(names.lon),
    'SensorZenith': str(names.vza) + '_tie',
    'SensorAzimuth': str(names.vaa) + '_tie',
    'SolarZenith': str(names.sza) + '_tie',
    'SolarAzimuth': str(names.saa) + '_tie',
}


def _as_floats(value) -> np.ndarray:
    """Coerce an HDF4 attribute value (number, list or string) to float64."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype='f8').ravel()
    s = str(value).strip()
    if s[:1] in '[(':
        s = s[1:-1]
    return np.array([float(t) for t in s.split(',') if t.strip()], dtype='f8')


class _ModisBase(Reader):
    """Shared implementation of the MODIS L1B family.

    A granule is a single HDF4 file whose payload is four radiance blocks
    (250 m, 500 m, 1 km reflective, 1 km emissive). The variant is detected
    from the INVENTORYMETADATA SHORTNAME in the merged Core/Archive/Struct
    metadata tree.
    """

    sensor = 'MODIS'
    chunks_default = 100

    # Per-level hooks, set by the subclasses ---------------------------------
    level = None
    payload_var = None
    required_vars = ()
    flags = True
    VARIANTS = {}

    # ------------------------------------------------------------------- open

    def _open(self, filepath: Path) -> xr.Dataset:
        """Open the HDF4 scene lazily (dask-backed); loud pyhdf fallback when netCDF4 lacks HDF4.

        The legacy reader's fallback was a dead branch (its error check was
        always true and unconditionally raised ImportError); here the -128
        'HDF4 unsupported' errno is matched explicitly and anything else is
        re-raised.

        h5py is imported before the netCDF4 attempt on purpose: if libnetcdf's
        copy of the HDF5 C library reaches the process first, it interposes
        with h5py's bundled libhdf5 (dynamic symbol resolution), and every
        later h5netcdf write in that process fails with a spurious
        'H5DSis_scale' error. Loading h5py first makes the fallback safe to
        follow with an HDF5-backed save.
        """
        if not filepath.is_file():
            raise ProductError(f'{self.sensor} {self.level}: no such file {filepath}')
        import h5py  # noqa: F401  (load order guard, see docstring)
        try:
            # chunks='auto' keeps every variable dask-backed from the open; an
            # unchunked netCDF4 read would materialize the whole granule in RAM
            # at construction (the per-block radiometry below, and the
            # da.full(..., chunks=fdata.chunks) calls, assume dask arrays).
            # The final spatial chunking is set in read() via self.chunks.
            return xr.open_dataset(filepath, engine='netcdf4', decode_cf=False,
                                   chunks='auto')
        except OSError as e:
            if '-128' not in str(e):
                raise
            log.warning(
                f'{self.sensor} {self.level}: the netCDF4 stack lacks HDF4 '
                f'support; falling back to pyhdf (eoread.hdf4.load_hdf4)')
            return load_hdf4(filepath, lazy=True, chunks=self.chunks_default)

    # ------------------------------------------------------------- metadata

    @staticmethod
    def _metadata(ds: xr.Dataset) -> dict:
        """Merge the three HDF-EOS text blocks into one header tree."""
        header = {}
        for block in ('CoreMetadata', 'ArchiveMetadata', 'StructMetadata'):
            key = block + '.0'
            if key in ds.attrs:
                header.update(parse_hdf4_metadata(ds.attrs[key]))
        return header

    @staticmethod
    def _acquisition_datetime(filepath: Path) -> str:
        """Acquisition start (UTC) from the granule name, ``.A<yyyy><ddd>.<hhmm>``.

        The file metadata only carries a time-of-day range (RANGEDATETIME) and
        the production datetime; the acquisition date comes from the filename.
        """
        m = re.search(r'\.A(\d{4})(\d{3})\.(\d{2})(\d{2})', filepath.name)
        if not m:
            raise ProductError(
                f'cannot parse the acquisition datetime from {filepath.name!r}')
        year, doy, hour, minute = (int(g) for g in m.groups())
        dt = datetime(year, 1, 1, hour, minute, tzinfo=timezone.utc) \
            + timedelta(days=doy - 1)
        return dt.isoformat()

    # ------------------------------------------------------------------ read

    def read(self, filepath: Path) -> xr.Dataset:
        if self.verbose:
            log.debug(f'Reading MODIS {self.level} product {filepath.name}')

        ds = self._open(filepath)

        rows, cols = str(names.rows), str(names.columns)
        ty, tx = rows + '_tie', cols + '_tie'
        bands = str(names.bands)

        # 0. Canonical names: strip the ':MODIS_SWATH_Type_L1B' scale suffix,
        # then map dims and variables to the registry names ------------------
        ds = ds.rename({d: _DIM_MAP[d.split(':')[0]]
                        for d in ds.dims if d.split(':')[0] in _DIM_MAP})
        ds = ds.rename({v: _VAR_MAP[v] for v in ds.data_vars if v in _VAR_MAP})

        # 1. Variant detection on the merged metadata tree ---------------------
        header = self._metadata(ds)
        spec = self.VARIANTS[self.detect_variant(header)]

        # 2. Explicit scaling (identical on both engine paths) -----------------
        # lat/lon: degrees, fill -999 -> NaN
        for var in (str(names.lat), str(names.lon)):
            a = ds[var]
            fill = a.attrs.pop('_FillValue', None)
            if fill is not None:
                ds[var] = a.where(a != float(fill))

        # tie-point angles: int16 * scale_factor, fill -32767 -> NaN
        for var in (str(names.vza) + '_tie', str(names.vaa) + '_tie',
                    str(names.sza) + '_tie', str(names.saa) + '_tie'):
            a = ds[var]
            scale = float(a.attrs.pop('scale_factor', 0.01))
            fill = a.attrs.pop('_FillValue', None)
            if fill is not None:
                a = a.where(a != float(fill))
            ds[var] = (a * scale).astype('float32')

        # gflags: kept raw uint8; the all-255 value is a valid all-bits-set
        # flag pattern, so no masking (documented break from v1)
        ds['gflags'].attrs.pop('_FillValue', None)

        # 3. Radiometry ----------------------------------------------------------
        # Ltoa = scale * (raw - offset); Rtoa from the file-native per-band
        # reflectance calibration (absent on the emissive block -> NaN). The
        # per-block arithmetic is done on the dask arrays directly: the band
        # factors broadcast along axis 0 and the string band labels are
        # assigned once after the concat, so no index alignment is involved.
        ltoa_parts, rtoa_parts = [], []
        labels_all, groups_all, units = [], [], None
        for var in _EVBLOCKS:
            raw = ds[var]
            bdim = next(d for d in raw.dims if d not in (rows, cols))
            f = raw.astype('float32')
            fill = raw.attrs.get('_FillValue', None)
            if fill is not None:
                f = f.where(f != float(fill))
            fdata = f.data

            sc = _as_floats(raw.attrs['radiance_scales']).reshape(-1, 1, 1)
            of = _as_floats(raw.attrs['radiance_offsets']).reshape(-1, 1, 1)
            ltoa_parts.append(xr.DataArray(
                ((fdata - of) * sc).astype('float32'), dims=(bands, rows, cols)))

            if 'reflectance_scales' in raw.attrs:
                rsc = _as_floats(raw.attrs['reflectance_scales']).reshape(-1, 1, 1)
                rof = _as_floats(raw.attrs['reflectance_offsets']).reshape(-1, 1, 1)
                rtoa_parts.append(xr.DataArray(
                    ((fdata - rof) * rsc).astype('float32'),
                    dims=(bands, rows, cols)))
            else:
                rtoa_parts.append(xr.DataArray(
                    da.full(fdata.shape, np.nan, dtype='float32',
                            chunks=fdata.chunks),
                    dims=(bands, rows, cols)))

            bvals = ds[bdim].values
            labels = [str(float(v)) for v in bvals]
            labels_all += labels
            groups_all += [bdim] * len(labels)
            units = units or raw.attrs.get('radiance_units')

        ds[str(names.ltoa)] = xr.concat(ltoa_parts, dim=bands)
        ds[str(names.rtoa)] = xr.concat(rtoa_parts, dim=bands)
        # Radiance unit only when the EV blocks state one (dimensionless Rtoa
        # gets no unit attribute at all).
        if units:
            ds[str(names.ltoa)].attrs['unit'] = str(units)

        # Public band labels: MODIS band numbers as 'B<N>'; the high-gain TDI
        # pairs 13.5/14.5 become B13HG/B14HG (the file-native values remain
        # the internal _CWVL lookup keys). Bands are sorted by ascending
        # central wavelength (file block order is not wavelength-ordered);
        # the stable sort keeps each 13/B13HG and 14/B14HG pair adjacent.
        pub = {'13.5': 'B13HG', '14.5': 'B14HG'}
        labels_pub = [pub.get(l, f'B{int(float(l))}') for l in labels_all]
        cw_arr = np.array([_CWVL[l] for l in labels_all], 'f8')
        order = np.argsort(cw_arr, kind='stable')
        ds[str(names.ltoa)] = ds[str(names.ltoa)].isel({bands: order})
        ds[str(names.rtoa)] = ds[str(names.rtoa)].isel({bands: order})
        ds = ds.assign_coords({
            bands: [labels_pub[i] for i in order],
            str(names.cwav): (bands, cw_arr[order]),
            str(names.bgroup): (bands, [groups_all[i] for i in order]),
        })
        ds[str(names.cwav)].attrs['unit'] = 'nm'

        # 4. Pixel geolocation ------------------------------------------------------
        # The 25 km tie grid maps onto the pixel grid with an exact integer
        # rule (data index i <-> geo index clip((i - 2) // 5, 0, n_geo - 1);
        # the scan axis is one pixel short of a multiple of five).
        ny, nx = ds.sizes[rows], ds.sizes[cols]
        iy = np.clip((np.arange(ny) - 2) // 5, 0, ds.sizes[ty] - 1)
        ix = np.clip((np.arange(nx) - 2) // 5, 0, ds.sizes[tx] - 1)
        for var in (str(names.lat), str(names.lon), 'gflags'):
            ds[var] = ds[var].isel({ty: iy, tx: ix}).rename({ty: rows, tx: cols})

        # 5. Raw-block cleanup ---------------------------------------------------------
        keep = {str(names.lat), str(names.lon), 'gflags',
                str(names.vza) + '_tie', str(names.vaa) + '_tie',
                str(names.sza) + '_tie', str(names.saa) + '_tie',
                str(names.ltoa), str(names.rtoa)}
        drop = [v for v in ds.data_vars if v not in keep]
        drop += [b for b in _DIM_MAP.values() if b not in keep and b in ds.variables]
        ds = ds.drop_vars(drop)

        # 6. Chunking, orientation -------------------------------------------------------
        nc = {rows: self.chunks[rows], cols: self.chunks[cols]}
        ds = ds.chunk(nc)
        ds = self._orient(ds)
        # The per-block payload concat leaves the bands dim chunked by block
        # while the geolocation upsample keeps other boundaries; unify so every
        # dask variable reports consistent chunk boundaries (Dataset.chunks).
        ds = ds.unify_chunks()

        # 7. Attributes --------------------------------------------------------------------
        inv = header['INVENTORYMETADATA']
        container = inv['ASSOCIATEDPLATFORMINSTRUMENTSENSOR'][
            'ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']
        rect = header['ARCHIVEDMETADATA']['BOUNDINGRECTANGLE']
        raw_attrs = dict(ds.attrs)
        ds.attrs = {}
        filter_fn = (lambda x, y: x) if self.metadata_template is None \
            else filter_metadata
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): str(container['ASSOCIATEDPLATFORMSHORTNAME']['VALUE']),
            str(names.product_name): str(inv['ECSDATAGRANULE']['LOCALGRANULEID']['VALUE']),
            str(names.shortname): str(inv['COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE']),
            str(names.input_directory): str(filepath.parent),
            str(names.datetime): self._acquisition_datetime(filepath),
            str(names.resolution): spec['resolution'],
            str(names.crs): 'EPSG:4326',
            'footprint_lat': (float(rect['NORTHBOUNDINGCOORDINATE']['VALUE']),
                              float(rect['SOUTHBOUNDINGCOORDINATE']['VALUE'])),
            'footprint_lon': (float(rect['EASTBOUNDINGCOORDINATE']['VALUE']),
                              float(rect['WESTBOUNDINGCOORDINATE']['VALUE'])),
            'version': int(inv['COLLECTIONDESCRIPTIONCLASS']['VERSIONID']['VALUE']),
            'night': str(inv['ECSDATAGRANULE']['DAYNIGHTFLAG']['VALUE']).lower() != 'day',
            'user_guide': _USER_GUIDE,
            '_srf_getter': 'eotools.srf.get_SRF_eumetsat',
            '_srf_getter_arg': 'eos_1_modis',
        })
        header['attributes'] = raw_attrs
        ds.attrs['metadata'] = filter_fn(header, self.metadata_template)

        return self.finalize(ds)

    # ------------------------------------------------------------- orientation

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

    # ----------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return ['gflags']

    @classmethod
    def flag_dims_like(cls) -> str:
        return 'gflags'

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """`gflags` is raw (the file carries no bit definitions): any bit set."""
        if flag_name == GenericFlags.L1_INVALID:
            return ds['gflags'] != 0
        raise ValueError(f'Unsupported MODIS flag: {flag_name!r}')


class Level1B_MODIS(_ModisBase):
    """MODIS L1B swath product (MOD021KM / MYD021KM).

    Payload `Ltoa` (TOA radiance, file units W.m-2.um-1.sr-1) on the 38
    file-native bands; declared extra `Rtoa` from the file's per-band
    reflectance calibration (NaN on the emissive block, which carries none).
    Quality flags decode through `Level1B_MODIS.getflag` (L1_INVALID = any bit
    set in the raw `gflags`).
    """

    level = 'L1B'
    payload_var = str(names.ltoa)
    sample_products = ('MODIS_L1_AQUA',)
    required_vars = (str(names.rtoa), 'gflags')

    #: L1B shortnames, keyed for `detect_variant` on the INVENTORYMETADATA
    #: SHORTNAME; an unknown value raises `UnsupportedProductType` at open
    #: time. Only Aqua is verified against a real sample (Terra follows the
    #: same layout).
    VARIANTS = {
        shortname: dict(
            detect={'field': ('INVENTORYMETADATA', 'COLLECTIONDESCRIPTIONCLASS',
                              'SHORTNAME', 'VALUE'),
                    'value': shortname},
            resolution=1000,
        )
        for shortname in ('MOD021KM', 'MYD021KM')
    }
