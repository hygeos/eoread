#!/usr/bin/env python3
"""
Sentinel-3 OLCI (Ocean and Land Colour Instrument) readers, levels 1 and 2.

One class per level over a shared sensor base (REFACTORING.md, §4):

    >>> from eoread.readers.olci import Level1_OLCI        # doctest: +SKIP
    >>> reader = Level1_OLCI('S3A_OL_1_EFR____....SEN3')   # doctest: +SKIP
    >>> ds = reader.ds                                     # doctest: +SKIP

Level 1 (EFR) carries TOA radiance for the 21 acquisition bands; `Rtoa` is a
declared extra computed lazily from the payload, the tie-point solar zenith
angle and the per-pixel solar flux. Level 2 (WFR/WRR water products) carries
water-leaving reflectance `rho_w` on the 16-band ocean subset plus the
log10-scaled uncertainty and the derived OC products; it has no flag reader
(the WQSF bit definitions are not in the core registry).

Differences from the legacy v1 reader (documented clean breaks, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the granule (canonical
  orientation). The native orientation is decided per file by a two-pixel
  latitude/longitude gradient check at the grid center.
- `latitude`/`longitude` are float32 per the naming registry; legacy emitted
  float64 and leaked the geo file's global attributes (subsampling factors)
  onto the variable attributes as a side effect. The subsampling factors are
  now read from the geo file *global* attributes only.
- The band table is built from the manifest `bandDescriptions` and cross-checked
  against the band files actually present on disk; any mismatch raises at open
  time with the full list of discrepancies (legacy failed later, mid per-band
  loop, with a bare FileNotFoundError).
- The instrument-data bands are selected positionally from the unindexed bands
  dimension of `instrument_data.nc` with an explicit bounds check, then
  re-labelled to the payload band table (legacy used a blind
  `int(band[2:]) - 1` selection).
- The nominal `resolution` attribute is per product type (EFR/WFR: 300 m,
  WRR: 600 m); legacy hard-coded 500 for every product.
- The L1 payload keeps the true file units (mW.m-2.sr-1.nm-1); legacy reported
  'W/sr/m^2'. `Rtoa` is a declared extra kept lazy, not materialized at read
  time.
- The optional `interp_angles` ('atan2'/'legacy') is dropped: angle
  interpolation uses the shared blockwise-linear transform
  (`eoread.transforms.angles.tiepoint_angles`).
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from core import log
from core.geo.naming import names
from core.table import read_xml
from core.tools import getflag

from eoread.flags import FlagsReaderBase, GenericFlags
from eoread.reader import ProductError, Reader, UnsupportedProductType
from eoread.transforms.angles import tiepoint_angles
from eoread.tools import filter_metadata, spatial_resample


#: Native spatial dim names of the OLCI NetCDF files, before rename to the
#: canonical grid dims.
_GRID_RENAME = {'rows': str(names.rows), 'columns': str(names.columns)}

#: Angle variables of tie_geometries.nc -> `*_tie` names expected by
#: `tiepoint_angles` (which adds the full-grid canonical variables).
_ANGLE_MAP = {
    'SZA': str(names.sza) + '_tie',
    'SAA': str(names.saa) + '_tie',
    'OZA': str(names.vza) + '_tie',
    'OAA': str(names.vaa) + '_tie',
}

#: Tie-point dim names -> canonical grid dim names (meteo resample outputs).
_TIE_GRID_RENAME = {'tie_rows': str(names.rows), 'tie_columns': str(names.columns)}

#: Meteorological tie variables of tie_meteo.nc -> canonical output names.
_METEO_MAP = (
    ('humidity', 'humidity'),
    ('sea_level_pressure', 'sea_level_pressure'),
    ('total_columnar_water_vapour', 'total_columnar_water_vapour'),
    ('total_ozone', 'total_column_ozone'),
)


class _OlciBase(Reader):
    """Shared implementation of the OLCI .SEN3 product family.

    A granule is a `.SEN3` directory (optionally nested one level deeper under
    a folder of the same name) holding an `xfdumanifest.xml` plus per-band and
    ancillary NetCDF files with native 'rows'/'columns' dims. The variant
    (EFR / WFR / WRR) is detected from the manifest
    `generalProductInformation/productType`; the product level is cross-checked
    against the content-unit `textInfo`.

    Per-level hooks set by the subclasses: `level`, `payload_var`,
    `_band_suffix` ('radiance' or 'reflectance'), `flags`, `required_vars`,
    `VARIANTS` and `_level_token` (the 'Level N' string expected in textInfo).
    """

    sensor = 'OLCI'
    chunks_default = 500

    # Per-level hooks, set by the subclasses ---------------------------------
    level = None
    payload_var = None
    required_vars = ()
    flags = None
    VARIANTS = {}
    _band_suffix = None
    _level_token = None

    def read(self, path: Path) -> xr.Dataset:
        dirname = Path(path)
        if not dirname.exists():
            raise ProductError(f'{self.sensor} {self.level}: {dirname} does not exist')
        # .SEN3 containers store the granule in a nested folder of the same name
        inner = dirname / dirname.name
        if inner.is_dir():
            dirname = inner
        manifest_file = dirname / 'xfdumanifest.xml'
        if not manifest_file.exists():
            raise ProductError(
                f'{self.sensor} {self.level}: no xfdumanifest.xml in {dirname}; '
                f'not an OLCI .SEN3 product')

        # 0. Manifest: header, level check, variant ---------------------------
        if self.verbose:
            log.debug(f'Reading OLCI {self.level} product {dirname.name}')
        manifest = read_xml(manifest_file)
        header = self._manifest_header(manifest)
        self._check_level(header['textInfo'])
        spec = self.VARIANTS[self.detect_variant(header)]

        # 1. Band table: manifest bandDescriptions x files on disk -------------
        labels, cwvl, files = self._band_table(manifest, dirname)

        chunks = self.chunks
        nc = {'rows': chunks[str(names.rows)], 'columns': chunks[str(names.columns)]}

        ds = xr.Dataset()
        ds = self._read_payload(ds, dirname, labels, files, nc)
        ds = ds.assign_coords({
            str(names.bands): labels,
            str(names.cwav): (str(names.bands), cwvl),
            str(names.bgroup): (str(names.bands), ['bands_vnir'] * len(labels)),
        })

        # 2. Geolocation --------------------------------------------------------
        # All opens pin engine='h5netcdf': the netCDF4 stack is broken in the
        # supported environments, and co-loading its bundled libhdf5 copy
        # breaks h5netcdf dimension-scale writes (see REFACTORING.md).
        geo = xr.open_dataset(dirname / 'geo_coordinates.nc', chunks=nc, engine='h5netcdf').rename(_GRID_RENAME)
        ac_factor = int(geo.attrs['ac_subsampling_factor'])
        al_factor = int(geo.attrs['al_subsampling_factor'])
        ds[str(names.lat)] = geo['latitude'].astype('float32')
        ds[str(names.lon)] = geo['longitude'].astype('float32')
        ds['altitude'] = geo['altitude']

        # 3. Tie-point angles (full grid, lazy blockwise interpolation) --------
        if self.verbose:
            log.debug('Read geometric tie points')
        ds = self._tie_points(ds, dirname / 'tie_geometries.nc', ac_factor, al_factor, _ANGLE_MAP)
        ds = tiepoint_angles(ds, chunks=chunks)

        # 4. Meteorological tie points ------------------------------------------
        if self.verbose:
            log.debug('Read meteorological tie points')
        ds = self._meteo(ds, dirname / 'tie_meteo.nc', ac_factor, al_factor, chunks)

        # 5. Instrument data (spectral tables + detector assignment) -----------
        ds = self._instrument_data(ds, dirname, labels, nc)

        # 6. Per-pixel spectral quantities (wavelength, solar flux) -------------
        ds[str(names.wav)] = self._per_pixel(ds['lambda0'], ds['detector_index'])
        ds[str(names.wav)].attrs.update(ds['lambda0'].attrs)
        ds[str(names.F0)] = self._per_pixel(ds['solar_flux'], ds['detector_index'])
        ds[str(names.F0)].attrs.update(ds['solar_flux'].attrs)

        # 7. Level-specific layers ----------------------------------------------
        ds = self._level_layers(ds, dirname, nc, header['platform'])

        # 8. Orientation: canonical south-up, west-to-east ----------------------
        ds = self._orient(ds)
        # The per-band payload concat leaves the bands dim as one chunk per
        # band while the instrument tables stay single-block; unify so every
        # dask variable reports consistent chunk boundaries (Dataset.chunks).
        ds = ds.unify_chunks()

        # 9. Attributes -----------------------------------------------------------
        filter_fn = (lambda x, y: x) if self.metadata_template is None \
            else filter_metadata
        ds.attrs.update({
            str(names.sensor): header['sensor'],
            str(names.platform): header['platform'],
            str(names.product_name): dirname.name,
            str(names.input_directory): str(dirname.parent),
            str(names.datetime): header['datetime'],
            str(names.resolution): spec['resolution'],
            str(names.shortname): header['productType'],
            str(names.crs): 'EPSG:4326',
            'footprint_lat': header['footprint_lat'],
            'footprint_lon': header['footprint_lon'],
            'metadata': filter_fn(manifest, self.metadata_template),
        })
        return ds

    # ------------------------------------------------------------- manifest

    @staticmethod
    def _manifest_header(manifest: dict) -> dict:
        """Extract the identification fields used for detection and attrs."""
        objs = {o['attributes']['ID']: o
                for o in manifest['metadataSection']['metadataObject']}
        textinfo = manifest['informationPackageMap']['contentUnit']['attributes']['textInfo']
        gpi = objs['generalProductInformation']['metadataWrap']['xmlData']['generalProductInformation']
        plat = objs['platform']['metadataWrap']['xmlData']['platform']
        acq = objs['acquisitionPeriod']['metadataWrap']['xmlData']['acquisitionPeriod']

        start = datetime.fromisoformat(acq['startTime'])
        stop = datetime.fromisoformat(acq['stopTime'])

        # Footprint: 'lat lon lat lon ...' pairs (legacy semantics)
        it = iter(objs['measurementFrameSet']['metadataWrap']['xmlData']
                  ['frameSet']['footPrint']['posList'].split())
        points = [(float(a), float(b)) for a, b in zip(it, it)]

        return {
            'textInfo': textinfo,
            'productType': gpi['productType'],
            'sensor': plat['instrument']['familyName']['attributes']['abbreviation'],
            'platform': plat['familyName'] + plat['number'],
            'datetime': (start + (stop - start) / 2.).isoformat(),
            'footprint_lat': tuple(p[0] for p in points),
            'footprint_lon': tuple(p[1] for p in points),
        }

    def _check_level(self, textinfo: str) -> None:
        """Loud level check against the content-unit textInfo."""
        found = re.findall(r'Level \d', textinfo)
        if len(found) != 1 or found[0] != self._level_token:
            raise UnsupportedProductType(
                f'{self.sensor} {self.level}: manifest textInfo is {textinfo!r}, '
                f'expected a {self._level_token!s} product')

    def _band_table(self, manifest: dict, dirname: Path) -> tuple[list, list, dict]:
        """Build the band table from the manifest and cross-check the files.

        Returns (labels, central_wavelengths, {label: file_path}) in manifest
        order; raises `ProductError` listing every discrepancy when the band
        files on disk do not match the manifest `bandDescriptions`.
        """
        info = next(o for o in manifest['metadataSection']['metadataObject']
                    if o['attributes']['ID'] == 'olciProductInformation')
        desc = info['metadataWrap']['xmlData']['olciProductInformation']['bandDescriptions']
        labels = [k for k in desc if k != 'attributes']
        cwvl = [float(desc[k]['centralWavelength']) for k in labels]

        suffix = f'_{self._band_suffix}.nc'
        files = {p.name[:-len(suffix)]: p
                 for p in dirname.glob(f'Oa*{suffix}')}
        missing = [b for b in labels if b not in files]
        extra = sorted(set(files) - set(labels))
        if missing or extra:
            raise ProductError(
                f'{self.sensor} {self.level}: band table mismatch between the '
                f'manifest bandDescriptions and the files on disk\n'
                f'  bands without file: {missing}\n'
                f'  files without band: {extra}')
        return labels, cwvl, files

    # --------------------------------------------------------------- payload

    def _read_payload(self, ds: xr.Dataset, dirname: Path, labels: list,
                      files: dict, nc: dict) -> xr.Dataset:
        """Read the per-band payload rasters (lazy, native grid dims)."""
        raise NotImplementedError

    # ------------------------------------------------------------ tie points

    def _tie_points(self, ds: xr.Dataset, path: Path, ac_factor: int,
                    al_factor: int, var_map: dict) -> xr.Dataset:
        """Open a tie-point file, align it to the grid and merge it into `ds`.

        The file's unindexed tie_rows/tie_columns dims are given pixel-position
        coordinates (arange x subsampling factor); the grid/tie alignment is
        checked loudly. `var_map` renames the file variables on the way in.
        """
        tie = xr.open_dataset(path, engine='h5netcdf').rename(var_map)
        tie = tie.assign_coords(
            tie_rows=np.arange(tie.sizes['tie_rows']) * al_factor,
            tie_columns=np.arange(tie.sizes['tie_columns']) * ac_factor,
        )
        ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
        if (nx - 1) != ac_factor * (tie.sizes['tie_columns'] - 1) \
                or (ny - 1) != al_factor * (tie.sizes['tie_rows'] - 1):
            raise ProductError(
                f'{self.sensor} {self.level}: tie-point grid '
                f"({tie.sizes['tie_rows']} x {tie.sizes['tie_columns']}, factors "
                f"ac={ac_factor}/al={al_factor}) does not span the data grid "
                f'({ny} x {nx})')
        return ds.merge(tie)

    def _meteo(self, ds: xr.Dataset, path: Path, ac_factor: int, al_factor: int,
               chunks: dict) -> xr.Dataset:
        """Interpolate the meteorological tie points to the full grid (lazy)."""
        tie = xr.open_dataset(path, engine='h5netcdf')
        tie = tie.assign_coords(
            tie_rows=np.arange(tie.sizes['tie_rows']) * al_factor,
            tie_columns=np.arange(tie.sizes['tie_columns']) * ac_factor,
        )
        ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
        shape = {'tie_rows': ny, 'tie_columns': nx}
        tie_chunks = {'tie_rows': chunks[str(names.rows)],
                      'tie_columns': chunks[str(names.columns)]}

        # Wind magnitude from the two vector components (no tie variable kept)
        wind0 = spatial_resample(tie['horizontal_wind'].isel(wind_vectors=0),
                                 shape, tie_chunks, 'linear')
        wind1 = spatial_resample(tie['horizontal_wind'].isel(wind_vectors=1),
                                 shape, tie_chunks, 'linear')
        ds['horizontal_wind'] = (
            np.sqrt(wind0**2 + wind1**2)
            .rename(_TIE_GRID_RENAME).transpose(str(names.rows), str(names.columns)))
        ds['horizontal_wind'].attrs = tie['horizontal_wind'].attrs

        for var_from, var_to in _METEO_MAP:
            raster = spatial_resample(tie[var_from], shape, tie_chunks, 'linear')
            ds[var_to] = (raster.rename(_TIE_GRID_RENAME)
                          .transpose(str(names.rows), str(names.columns)))
            ds[var_to].attrs = tie[var_from].attrs
            ds[f'{var_to}_tie'] = tie[var_from]
        return ds

    # -------------------------------------------------------- instrument data

    def _instrument_data(self, ds: xr.Dataset, dirname: Path, labels: list,
                         nc: dict) -> xr.Dataset:
        """Read the spectral tables and the per-pixel detector assignment.

        `instrument_data.nc` carries an *unindexed* bands dimension of 21
        entries (Oa01..Oa21 positions); the subset relevant to this product is
        selected positionally with a bounds check, then re-labelled to the
        payload band table so it merges without coordinate conflicts.
        """
        inst = xr.open_dataset(
            dirname / 'instrument_data.nc',
            mask_and_scale=False,
            # duplicate-dimension variable, not part of the output
            drop_variables='relative_spectral_covariance',
            chunks=nc,
            engine='h5netcdf',
        ).rename(_GRID_RENAME)
        index = [int(b[2:]) - 1 for b in labels]
        if any(i < 0 or i >= inst.sizes[str(names.bands)] for i in index):
            raise ProductError(
                f'{self.sensor} {self.level}: band table {labels} does not map '
                f'onto the instrument_data bands dimension '
                f'(size {inst.sizes[str(names.bands)]})')
        inst = inst.isel({str(names.bands): index}).assign_coords(
            {str(names.bands): labels})
        for var in ('FWHM', 'detector_index', 'frame_offset', 'lambda0', 'solar_flux'):
            ds[var] = inst[var]
        return ds

    @staticmethod
    def _per_pixel(table: xr.DataArray, detector_index: xr.DataArray) -> xr.DataArray:
        """Gather the per-detector value of a (bands, detectors) table for each pixel."""
        return xr.apply_ufunc(
            lambda t, di: t[:, 0, 0, di],
            table,
            detector_index,
            dask='parallelized',
            input_core_dims=[['detectors'], []],
            output_dtypes=[table.dtype],
        )

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

    # -------------------------------------------------------- level-specific

    def _level_layers(self, ds: xr.Dataset, dirname: Path, nc: dict,
                      platform: str) -> xr.Dataset:
        """Read the level-specific ancillary layers (per-level hook)."""
        raise NotImplementedError


class Level1_OLCI(_OlciBase):
    """Sentinel-3 OLCI level-1 full-resolution (EFR) product, 300 m.

    Payload `Ltoa` (TOA radiance, true file units mW.m-2.sr-1.nm-1) on the 21
    acquisition bands; declared extra `Rtoa` = pi * Ltoa / (cos(sza) * F0),
    computed lazily from the tie-point solar zenith angle and the per-pixel
    solar flux. Quality flags decode through `FlagsReader_OLCI` (LAND and
    L1_INVALID from the `quality_flags` bit field).
    """

    level = 'L1'
    payload_var = str(names.ltoa)
    sample_products = ('OLCI_L1_EFR',)
    required_vars = (str(names.rtoa), 'quality_flags')
    flags = 'eoread.readers.olci.FlagsReader_OLCI'
    _band_suffix = 'radiance'
    _level_token = 'Level 1'

    #: Product types of the OLCI L1 family, keyed for `detect_variant`. Only
    #: entries verified against a real sample are listed; an unknown
    #: productType (e.g. an ERR granule) raises `UnsupportedProductType` at
    #: open time.
    VARIANTS = {
        'EFR': dict(
            detect={'field': 'productType', 'value': 'OL_1_EFR___'},
            resolution=300,
        ),
    }

    def _read_payload(self, ds: xr.Dataset, dirname: Path, labels: list,
                      files: dict, nc: dict) -> xr.Dataset:
        arrays = [xr.open_dataarray(files[b], chunks=nc, engine='h5netcdf')
                  .rename(_GRID_RENAME) for b in labels]
        ltoa = xr.concat(arrays, dim=str(names.bands))
        if 'units' not in arrays[0].attrs:
            raise ProductError(
                f'{self.sensor} {self.level}: radiance files carry no units '
                f'attribute; cannot report the payload unit')
        ds[str(names.ltoa)] = ltoa
        ds[str(names.ltoa)].attrs['unit'] = arrays[0].attrs['units']
        return ds

    def _level_layers(self, ds: xr.Dataset, dirname: Path, nc: dict,
                      platform: str) -> xr.Dataset:
        # Quality flags (decoded on demand via FlagsReader_OLCI)
        qf = xr.open_dataset(dirname / 'qualityFlags.nc', chunks=nc, engine='h5netcdf').rename(_GRID_RENAME)
        ds['quality_flags'] = qf['quality_flags']

        # Declared extra: TOA reflectance, kept lazy. Assigned as a plain
        # DataArray: in the (dims, data) tuple form xarray would try to wrap
        # the DataArray as Variable data and reject it.
        mus = np.cos(np.radians(ds[str(names.sza)]))
        rtoa = np.pi * ds[str(names.ltoa)] / (mus * ds[str(names.F0)])
        ds[str(names.rtoa)] = rtoa
        ds[str(names.rtoa)].attrs['unit'] = None

        # SRF getter (loud on unknown platforms)
        srf_arg = {
            'Sentinel-3A': 'sentinel3_1_olci',
            'Sentinel-3B': 'sentinel3_2_olci',
        }
        if platform not in srf_arg:
            raise ProductError(
                f'{self.sensor} {self.level}: no SRF getter argument for '
                f'platform {platform!r}')
        ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_olci'
        ds.attrs['_srf_getter_arg'] = srf_arg[platform]
        return ds


class Level2_OLCI(_OlciBase):
    """Sentinel-3 OLCI level-2 water product (WFR 300 m / WRR 600 m).

    Payload `rho_w` (water-leaving reflectance, dimensionless) on the 16-band
    ocean subset, with the log10-scaled `uncertainty` and the derived OC
    products (chl_nn, chl_oc4me, wqsf, A865, T865) as plain extras. No flag
    reader: the WQSF bit definitions are not in the core registry.
    """

    level = 'L2'
    payload_var = str(names.rho_w)
    sample_products = ('OLCI_L2_WFR', 'OLCI_L2_WRR')
    required_vars = ('uncertainty',)
    flags = None
    _band_suffix = 'reflectance'
    _level_token = 'Level 2'

    #: Product types of the OLCI L2 family, keyed for `detect_variant`. The
    #: manifest textInfo is identical for WFR and WRR ('Level 2 Water
    #: Product'), so the variant comes from the productType field. Only
    #: entries verified against a real sample are listed.
    VARIANTS = {
        'WFR': dict(
            detect={'field': 'productType', 'value': 'OL_2_WFR___'},
            resolution=300,
        ),
        'WRR': dict(
            detect={'field': 'productType', 'value': 'OL_2_WRR___'},
            resolution=600,
        ),
    }

    def _read_payload(self, ds: xr.Dataset, dirname: Path, labels: list,
                      files: dict, nc: dict) -> xr.Dataset:
        rhos, uncs = [], []
        for band in labels:
            f = xr.open_dataset(files[band], chunks=nc, engine='h5netcdf').rename(_GRID_RENAME)
            rhos.append(f[f'{band}_reflectance'])
            uncs.append(f[f'{band}_reflectance_unc'])
        ds[str(names.rho_w)] = xr.concat(rhos, dim=str(names.bands))
        # Dimensionless ratio: no units attribute in the files (MSI L2A precedent)
        ds[str(names.rho_w)].attrs['unit'] = None
        ds['uncertainty'] = xr.concat(uncs, dim=str(names.bands))
        return ds

    def _level_layers(self, ds: xr.Dataset, dirname: Path, nc: dict,
                      platform: str) -> xr.Dataset:
        for filename, var, out in (
                ('chl_nn.nc', 'CHL_NN', 'chl_nn'),
                ('chl_oc4me.nc', 'CHL_OC4ME', 'chl_oc4me'),
                ('wqsf.nc', 'WQSF', 'wqsf')):
            f = xr.open_dataset(dirname / filename, chunks=nc, engine='h5netcdf').rename(_GRID_RENAME)
            ds[out] = f[var]
        aer = xr.open_dataset(dirname / 'w_aer.nc', chunks=nc, engine='h5netcdf').rename(_GRID_RENAME)
        for var in ('A865', 'T865'):
            ds[var] = aer[var]
        return ds


class FlagsReader_OLCI(FlagsReaderBase):
    """Decode the generic flags of an OLCI level-1 product.

    Bit definitions come from the `flag_meanings` attribute of the
    `quality_flags` variable (core registry): LAND is 'land' minus
    'fresh_inland_water'; L1_INVALID is 'invalid'.
    """

    def requires(self) -> list[str]:
        return ['quality_flags']

    def dims_like(self) -> str:
        return 'quality_flags'

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.LAND:
            land = getflag(ds['quality_flags'], 'land')
            fresh_inland_water = getflag(ds['quality_flags'], 'fresh_inland_water')
            return land & ~fresh_inland_water
        if flag_name == GenericFlags.L1_INVALID:
            return getflag(ds['quality_flags'], 'invalid')
        raise ValueError(f'Unsupported flag: {flag_name}')
