#!/usr/bin/env python3
"""
ECOSTRESS readers.

Two PGE container families are handled by two sensor bases (one class per
level, REFACTORING.md §4):

- `Level1C_Ecostress` — HDF-EOS5 gridded radiance products.
- `Level1B_Ecostress`, `Level2_Ecostress` — plain-HDF5 swath products (the
  v001 L1B/L2 maquette and the v002 L2 LSTE product). These carry no
  per-pixel lat/lon: each reader pairs its input file with the sibling
  L1B_GEO file for the geolocation layers (see `_EcostressPlainBase`).

    >>> from eoread.readers.ecostress import Level1C_Ecostress  # doctest: +SKIP
    >>> reader = Level1C_Ecostress('ECOv002_L1CG_RAD_...h5')    # doctest: +SKIP
    >>> ds = reader.ds                                          # doctest: +SKIP

The ECOSTRESS level-1 radiance products of the PGE promise a single output
contract (TOA radiance + brightness temperature, water/cloud/data-quality
layers on one 70 m grid), so they are handled by one class with a variant
table keyed on the product ShortName; unknown types raise at open time.
Only variants verified against a real sample are listed — see VARIANTS.

Differences from the legacy v1 reader (documented in the REFACTORING.md
decision log):

- The output is south-up: row 0 is the southern edge of the granule
  (canonical orientation). Legacy output was north-up, which failed the
  `generic.Test.latlon` invariants.
- `latitude`/`longitude` are float32 per the naming registry; legacy
  emitted float64.
- The scene bounding box is parsed from the unfiltered StandardMetadata:
  the legacy reader read it from the *filtered* metadata dict and broke on
  minimal `metadata_template`.
- Flags decode with dtype-robust tests (`water == 0` instead of the legacy
  bitwise `~water`, which raised on this product's float64 water layer).
- `crs` is reported as 'EPSG:4326' (the product PROJ string
  '+proj=longlat +datum=WGS84' is WGS84 longlat).
"""

from pathlib import Path
from re import findall

import numpy as np
import xarray as xr

from core import log
from core.geo.naming import names
from core.tools import merge

from eoread.flags import FlagsReaderBase, GenericFlags
from eoread.reader import ProductError, Reader, UnsupportedProductType
from eoread.tools import filter_metadata
from eoread.transforms.geometry import latlon_bbox


#: ECOSTRESS L1 user guide (kept for legacy attribute parity)
USER_GUIDE = ('https://ecostress.jpl.nasa.gov/downloads/userguides/'
              '1_ECOSTRESS_L1_UserGuide_20190619.pdf')

#: Planck constants of the brightness-temperature inversion (µm, W/sr/m^2)
_K1 = 1.191042e8
_K2 = 1.4387752e4

#: HDF-EOS5 paths shared by every ECOSTRESS product
_STANDARD_METADATA = 'HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata'
_PRODUCT_METADATA = 'HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata'


class _EcostressBase(Reader):
    """Shared implementation of the ECOSTRESS level readers (HDF-EOS5)."""

    sensor = 'ECOSTRESS'
    chunks_default = 500
    flags = 'eoread.readers.ecostress.FlagsReader_ECOSTRESS'

    # Per-level hooks, set by the subclasses ---------------------------------
    level = None
    payload_var = None
    required_vars = ()

    def read(self, path: Path) -> xr.Dataset:
        h5 = Path(path).resolve()
        assert h5.exists(), f'{h5} does not exist'
        if self.verbose:
            log.debug(f'Reading ECOSTRESS {self.level} product {h5.name}')

        try:
            tree = xr.open_datatree(h5, phony_dims='sort', engine='h5netcdf')
        except Exception as exc:
            raise ProductError(
                f'{self.sensor} {self.level}: cannot open {h5.name} as an '
                f'HDF-EOS5 product ({type(exc).__name__}: {exc})') from exc

        try:
            standard = tree[_STANDARD_METADATA].to_dataset()
        except KeyError as exc:
            raise UnsupportedProductType(
                f'{self.sensor} {self.level}: {h5.name} carries no HDF-EOS5 '
                f'StandardMetadata node; not an ECOSTRESS product') from exc
        metadata = {k: standard[k].item() for k in standard.data_vars}

        instrument = metadata.get('InstrumentShortName')
        if instrument != self.sensor:
            raise UnsupportedProductType(
                f'{self.sensor} {self.level}: product is from sensor '
                f'{instrument!r}, not {self.sensor!r}')

        spec = self.VARIANTS[self.detect_variant(metadata)]

        # 1. Payload grid: rename the two spatial dims, keep it lazy ----------
        try:
            raw = tree[spec['grid']].to_dataset()
        except KeyError as exc:
            raise ProductError(
                f'{self.sensor} {self.level}: grid {spec["grid"]} not found '
                f'in {h5.name}') from exc
        nlines, npixels = int(metadata['ImageLines']), int(metadata['ImagePixels'])
        dims = list(raw.dims)
        assert len(dims) == 2 and raw.sizes[dims[0]] == nlines \
            and raw.sizes[dims[1]] == npixels, (
                f'{self.sensor} {self.level}: unexpected grid shape '
                f'{dict(raw.sizes)}, expected ({nlines}, {npixels})')
        raw = raw.rename({dims[0]: str(names.rows), dims[1]: str(names.columns)})
        raw = raw.chunk(chunks=self.chunks)

        # 2. Radiometry: merge per-band fields, calibrate, Planck -------------
        ds = merge(raw, dim=str(names.bands), pattern=r'(.+)_(\d+)')
        ds = ds.rename({'radiance': str(names.ltoa)})
        ds[str(names.ltoa)].attrs['unit'] = 'W/sr/m^2'

        product = tree[_PRODUCT_METADATA].to_dataset()
        cwvl_um = np.asarray(product['BandSpecification'].values).ravel()[1:]
        gain = np.asarray(product['CalibrationGainCorrection'].values).ravel()
        offset = np.asarray(product['CalibrationOffsetCorrection'].values).ravel()
        nbands = ds.sizes[str(names.bands)]
        assert len(cwvl_um) == nbands == len(gain) == len(offset), (
            f'{self.sensor} {self.level}: calibration table length mismatch '
            f'({len(cwvl_um)} wavelengths, {nbands} data bands, {len(gain)} '
            f'gains, {len(offset)} offsets)')

        ds[str(names.cwav)] = (str(names.bands), cwvl_um * 1e3)  # µm -> nm

        valid = ~ds[str(names.ltoa)].isnull()
        # The per-band calibration tables are broadcast along the leading band
        # axis; reshape to (n, 1, 1) so numpy aligns them with `bands` instead
        # of the trailing column axis.
        cwvl = cwvl_um.reshape(nbands, 1, 1)
        calibrated = gain.reshape(nbands, 1, 1) * ds[str(names.ltoa)].where(valid) \
            + offset.reshape(nbands, 1, 1)
        ds[str(names.bt)] = _K2 / (cwvl * np.log(
            _K1 / (calibrated.where(valid) * cwvl ** 5) + 1))
        ds[str(names.bt)].attrs['unit'] = 'Kelvin'

        # 3. Geolocation: the grid is axis-aligned in geographic space, so the
        #    scene bounding box pins the lat/lon rasters (north-up, as stored)
        north, south, east, west = self._scene_corners(metadata)
        ds = ds.assign(latlon_bbox(
            north, south, west, east,
            nx=ds.sizes[str(names.columns)], ny=ds.sizes[str(names.rows)],
            chunks=self.chunks))

        # 4. Orientation: the product grid is north-up (row 0 = high latitude);
        #    flip data and lat/lon together to the canonical south-up --------
        ds = ds.isel({str(names.rows): slice(None, None, -1)})
        ny, nx = ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]
        ds = ds.assign_coords({
            str(names.rows): np.arange(ny),
            str(names.columns): np.arange(nx),
        })

        # 5. Band labels: the integer suffixes become string coordinates ------
        ds = ds.assign_coords({
            str(names.bands): ds[str(names.bands)].values.astype(str),
            str(names.bgroup): (str(names.bands), ['bands_ir'] * nbands),
        })

        # 6. Attributes --------------------------------------------------------
        filter_fn = (lambda x, y: x) if self.metadata_template is None \
            else filter_metadata
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): metadata['PlatformShortName'],
            str(names.product_name): h5.name,
            str(names.input_directory): str(h5.parent),
            str(names.datetime): metadata['ProductionDateTime'],
            str(names.resolution): spec['resolution'],
            str(names.shortname): metadata['ShortName'],
            str(names.crs): 'EPSG:4326',
            'user_guide': USER_GUIDE,
            'daytime': bool(metadata['DayNightFlag'] == 'Day'),
            '_srf_getter': 'eotools.srf.get_SRF_eumetsat',
            '_srf_getter_arg': 'iss_1_ecostres',
            'metadata': filter_fn(metadata, self.metadata_template),
            'hdfeos_info': filter_fn(self._struct_metadata(tree),
                                     self.metadata_template),
        })
        return ds

    @staticmethod
    def _scene_corners(metadata: dict):
        """(north, south, east, west) in degrees from the scene boundary WKT."""
        points = _parse_wkt(metadata['SceneBoundaryLatLonWKT'])
        return (float(points[:, 1].max()), float(points[:, 1].min()),
                float(points[:, 0].max()), float(points[:, 0].min()))

    @staticmethod
    def _struct_metadata(tree: xr.DataTree) -> dict:
        """Parse the HDF-EOS5 StructMetadata text into a nested dict."""
        info = tree['HDFEOS INFORMATION']['StructMetadata.0'].values.item().decode()
        parser = _StructMetadataParser(info.split('\n'))
        parser.parse()
        return parser.data


class Level1C_Ecostress(_EcostressBase):
    """ECOSTRESS level-1C gridded radiance product (70 m grid).

    The PGE labels its radiance products 'L1C' + G/T: ECO_L1CG_RAD
    (granule) and ECO_L1CT_RAD (tile) — 'Level 1 Calibrated' radiance.
    The level label matches the legacy entry point `Level1C_ECOSTRESS`.
    """

    level = 'L1C'
    payload_var = str(names.ltoa)
    sample_products = ('ECOSTRESS_L1C',)
    required_vars = (str(names.bt), 'water', 'cloud', 'data_quality')

    #: Product types of the ECOSTRESS L1 family, keyed for `detect_variant`.
    #: Only entries verified against a real sample are listed; an unknown
    #: ShortName raises `UnsupportedProductType` at open time.
    VARIANTS = {
        'L1CG_RAD': dict(
            detect={'field': 'ShortName', 'value': 'ECO_L1CG_RAD'},
            grid='HDFEOS/GRIDS/ECO_L1CG_RAD_70m/Data Fields',
            resolution=70,
        ),
    }


class _StructMetadataParser:
    """Parser for HDF-EOS5 StructMetadata text (ported from the legacy reader)."""

    def __init__(self, text: list):
        self.data = {}
        self.text = text.copy()

    def empty(self) -> bool:
        return len(self.text) == 0

    def consume(self) -> str:
        line = self.text[0]
        if len(self.text) == 1:  # case for last line
            self.text = []
            return line.strip()

        self.text = self.text[1:]

        line = line.strip()
        while line == "":
            line = self.consume()

        return line.strip()

    def peek(self) -> str:
        return self.text[0].strip()

    def parse(self) -> None:
        while not self.empty():
            end = self._parse_recu(self.data)
            if end:
                break

    def _parse_recu(self, data: dict = None) -> bool:
        line = self.consume()
        if line == "END":
            return True

        key, val = line.split('=')
        if key in ['GROUP', 'OBJECT']:
            data[val] = {}

            line = self.peek()
            while f'END_' not in line:
                self._parse_recu(data[val])
                line = self.peek()  # refresh peeked line !!

            # closing tag
            self.consume()

        else:
            data[key] = val

        return False


def _parse_wkt(wkt: str) -> np.ndarray:
    """Parse a WKT polygon into an (n, 2) array of (lon, lat) degrees."""
    points = findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', wkt)
    points = [(float(lon), float(lat)) for lon, lat in points]
    return np.array(points)


class FlagsReader_ECOSTRESS(FlagsReaderBase):
    """
    Decode the generic flags of an ECOSTRESS L1 product from its data layers.

    Legacy semantics, made dtype-robust: this product stores water/cloud as
    float64, where the legacy bitwise `~water` raised.

    - L1_INVALID:   band 4 radiance is missing
    - LAND:         water mask is zero
    - CLOUD:        cloud layer is non-zero
    - L1_DEGRADED:  any data-quality layer is non-zero
    """

    def requires(self) -> list[str]:
        # Every variable used by getflag (the legacy reader under-declared
        # water, cloud and data_quality here).
        return ['water', 'cloud', 'data_quality', str(names.ltoa)]

    def dims_like(self) -> str:
        return 'water'

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.L1_INVALID:
            band = ds[str(names.ltoa)].sel({str(names.bands): '4'})
            return band.isnull()
        if flag_name == GenericFlags.LAND:
            return ds['water'] == 0
        if flag_name == GenericFlags.CLOUD:
            return ds['cloud'] != 0
        if flag_name == GenericFlags.L1_DEGRADED:
            quality = ds['data_quality'].max(dim=str(names.bands))
            return quality != 0
        raise ValueError(f'Unsupported ECOSTRESS flag: {flag_name!r}')

    def getflag_raw(self, ds: xr.Dataset, flag_name: str) -> xr.DataArray:
        raise NotImplementedError('ECOSTRESS flags are derived from data variables')


# ===========================================================================
# Plain-HDF5 family: v001 L1B/L2 maquette products + v002 L2 LSTE
# ===========================================================================


class _EcostressPlainBase(Reader):
    """Shared implementation of the plain-HDF5 ECOSTRESS products.

    The v001 L1B/L2 products (TRISHNA maquette) and the v002 L2 product are
    plain HDF5 rather than the HDF-EOS5 container of the gridded family: a
    `StandardMetadata` group, a product-specific metadata group and the data
    grid sit at the root. None of them carries per-pixel lat/lon, so each
    reader pairs its input file with the sibling geolocation (GEO) file in
    the same directory — the same name with the product token substituted
    (L1B_RAD -> L1B_GEO, L2_LSTE -> L1B_GEO).

    The PGE stores every granule in scan order and fully accounts for the
    instrument readout flip inside the GEO product (user guide §3.4: users of
    the radiance products need no special RadScanLineOrder handling), so a
    payload row/column pair addresses the same pixel as the GEO row/column
    pair. The canonical output is south-up, west-to-east; `_orient` decides
    the flip per file from a two-pixel latitude/longitude gradient check at
    the grid center (the swath is a tilted parallelogram, so corners are not
    reliable).
    """

    sensor = 'ECOSTRESS'
    chunks_default = 500

    # Per-level hooks, set by the subclasses ---------------------------------
    level = None
    payload_var = None
    required_vars = ()
    flags = None

    def read(self, path: Path) -> xr.Dataset:
        h5 = Path(path).resolve()
        assert h5.exists(), f'{h5} does not exist'
        if self.verbose:
            log.debug(f'Reading ECOSTRESS {self.level} product {h5.name}')

        try:
            tree = xr.open_datatree(h5, phony_dims='sort', engine='h5netcdf')
        except Exception as exc:
            raise ProductError(
                f'{self.sensor} {self.level}: cannot open {h5.name} as an '
                f'HDF5 product ({type(exc).__name__}: {exc})') from exc

        try:
            standard = tree['StandardMetadata'].to_dataset()
        except KeyError as exc:
            raise UnsupportedProductType(
                f'{self.sensor} {self.level}: {h5.name} carries no root '
                f'StandardMetadata group; not an ECOSTRESS product') from exc
        metadata = {k: standard[k].item() for k in standard.data_vars}

        instrument = metadata.get('InstrumentShortName')
        if instrument != self.sensor:
            raise UnsupportedProductType(
                f'{self.sensor} {self.level}: product is from sensor '
                f'{instrument!r}, not {self.sensor!r}')

        spec = self.VARIANTS[self.detect_variant(metadata)]
        if 'flags' in spec:
            # Per-variant flag reader (the v002 cloud mask has no v001
            # counterpart); the instance attribute shadows the class default
            # for finalize().
            self.flags = spec['flags']

        # 1. Payload grid: rename the two spatial dims, keep it lazy ----------
        try:
            raw = tree[spec['grid']].to_dataset()
        except KeyError as exc:
            raise ProductError(
                f'{self.sensor} {self.level}: grid {spec["grid"]} not found '
                f'in {h5.name}') from exc
        nlines, npixels = int(metadata['ImageLines']), int(metadata['ImagePixels'])
        dims = list(raw.dims)
        assert len(dims) == 2 and raw.sizes[dims[0]] == nlines \
            and raw.sizes[dims[1]] == npixels, (
                f'{self.sensor} {self.level}: unexpected grid shape '
                f'{dict(raw.sizes)}, expected ({nlines}, {npixels})')
        raw = raw.rename({dims[0]: str(names.rows), dims[1]: str(names.columns)})
        raw = raw.chunk(chunks=self.chunks)

        ds = self.payload(raw, tree, spec)

        # 2. Geolocation from the paired GEO file -----------------------------
        geo = self._geo_layers(h5, spec)
        assert geo.sizes[str(names.rows)] == ds.sizes[str(names.rows)] \
            and geo.sizes[str(names.columns)] == ds.sizes[str(names.columns)], (
                f'{self.sensor} {self.level}: GEO grid {dict(geo.sizes)} does '
                f'not match the payload grid {dict(ds.sizes)}')
        ds = ds.merge(geo)

        # 3. Orientation: canonical south-up, west-to-east --------------------
        ds = self._orient(ds)

        # 4. Band labels: the integer suffixes become string coordinates ------
        if str(names.bands) in ds.dims:
            nbands = ds.sizes[str(names.bands)]
            ds = ds.assign_coords({
                str(names.bands): ds[str(names.bands)].values.astype(str),
                str(names.bgroup): (str(names.bands), ['bands_ir'] * nbands),
            })

        # 5. Attributes --------------------------------------------------------
        filter_fn = (lambda x, y: x) if self.metadata_template is None \
            else filter_metadata
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): metadata['PlatformShortName'],
            str(names.product_name): h5.name,
            str(names.input_directory): str(h5.parent),
            str(names.datetime): metadata['ProductionDateTime'],
            str(names.resolution): spec['resolution'],
            str(names.shortname): metadata['ShortName'],
            str(names.crs): 'EPSG:4326',
            'user_guide': USER_GUIDE,
            'daytime': bool(metadata['DayNightFlag'] == 'Day'),
            '_srf_getter': 'eotools.srf.get_SRF_eumetsat',
            '_srf_getter_arg': 'iss_1_ecostres',
            'metadata': filter_fn(metadata, self.metadata_template),
        })
        return ds

    def payload(self, raw: xr.Dataset, tree: xr.DataTree, spec: dict) -> xr.Dataset:
        """Build the payload variables from the raw grid (per-level hook)."""
        raise NotImplementedError

    def _geo_layers(self, h5: Path, spec: dict) -> xr.Dataset:
        """Geolocation layers from the paired L1B_GEO file.

        The GEO file must sit next to the input file with the product token
        substituted in its name; it is not optional — these products carry no
        per-pixel lat/lon of their own.
        """
        from_token, to_token = spec['geo_token']
        geo_path = h5.with_name(h5.name.replace(from_token, to_token))
        if not geo_path.exists():
            raise ProductError(
                f'{self.sensor} {self.level}: paired geolocation file '
                f'{geo_path.name} not found next to {h5.name}; the ECOSTRESS '
                f'L1B/L2 products carry no per-pixel lat/lon and require it')

        tree = xr.open_datatree(geo_path, phony_dims='sort', engine='h5netcdf')
        try:
            geo = tree['Geolocation'].to_dataset()
        except KeyError as exc:
            raise ProductError(
                f'{self.sensor} {self.level}: {geo_path.name} carries no '
                f'Geolocation group') from exc

        dims = list(geo['latitude'].dims)
        geo = geo.rename({
            dims[0]: str(names.rows),
            dims[1]: str(names.columns),
            'latitude': str(names.lat),
            'longitude': str(names.lon),
            'solar_zenith': str(names.sza),
            'solar_azimuth': str(names.saa),
            'view_zenith': str(names.vza),
            'view_azimuth': str(names.vaa),
        })
        # height and line_start_time_j2000 are not part of the v2 contract.
        geo = geo.drop_vars([v for v in ('height', 'line_start_time_j2000')
                             if v in geo])

        # lat/lon float64 -> float32 per the naming registry (the angles are
        # already float32).
        geo[str(names.lat)] = geo[str(names.lat)].astype('float32')
        geo[str(names.lon)] = geo[str(names.lon)].astype('float32')

        # land_fraction uses a custom fill sentinel (-9999) rather than a CF
        # _FillValue, so h5netcdf leaves it raw: map the sentinel to NaN.
        land_fraction = geo['land_fraction']
        land_fraction = land_fraction.where(land_fraction != -9999)
        geo['land_fraction'] = land_fraction

        # Derived land/water layer: the v002 PGE water_mask is exactly
        # land_fraction < 50 (verified pixel-identical on a coastal scene),
        # so both variants share this single definition. Off-swath pixels
        # (NaN) count as land.
        geo['water'] = (land_fraction < 50).astype('uint8')
        return geo.chunk(chunks=self.chunks)

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


class Level1B_Ecostress(_EcostressPlainBase):
    """ECOSTRESS level-1B swath radiance product (v001 PGE, 70 m).

    'L1B' is the official v001-PGE family name (ECOSTRESS_L1B_RAD / _L1B_GEO)
    — see the REFACTORING.md decision log. The input is the RAD file; the
    paired L1B_GEO file must sit next to it and is auto-discovered by name
    substitution.

    The radiance_1..5 fields are already calibrated (TOA radiance in
    W/m^2/sr/µm — magnitudes invert to physical brightness temperatures), so
    `BT` is a direct Planck inversion without a gain/offset step. `cwav` comes
    from the product's own BandSpecification, whose band 0 holds the 1.6 µm NIR
    wavelength that is not part of this product: only the five thermal bands
    are stored.

    The GEO `height` and `line_start_time_j2000` fields are not part of the v2
    contract and are dropped.
    """

    level = 'L1B'
    payload_var = str(names.ltoa)
    sample_products = ('ECOSTRESS_L1B',)
    required_vars = (str(names.bt), 'data_quality', 'water', 'land_fraction')
    flags = 'eoread.readers.ecostress.FlagsReader_Level1B'

    #: Product types of the ECOSTRESS L1B family, keyed for `detect_variant`.
    #: Only entries verified against a real sample are listed; an unknown
    #: ShortName raises `UnsupportedProductType` at open time. (A v002 RAD
    #: product, if it carried this exact ShortName, would be read with the
    #: same layout — the two PGE generations are structurally identical.)
    VARIANTS = {
        'L1B_RAD': dict(
            detect={'field': 'ShortName', 'value': 'L1B_RAD'},
            grid='Radiance',
            product_metadata='L1B_RADMetadata',
            geo_token=('L1B_RAD', 'L1B_GEO'),
            resolution=70,
        ),
    }

    def payload(self, raw: xr.Dataset, tree: xr.DataTree, spec: dict) -> xr.Dataset:
        # radiance_1..5 / data_quality_1..5 merge into band stacks; anything
        # else in the grid is not part of the output contract.
        ds = merge(raw, dim=str(names.bands), pattern=r'(.+)_(\d+)')
        ds = ds.rename({'radiance': str(names.ltoa)})
        ds[str(names.ltoa)].attrs['unit'] = 'W/sr/m^2'
        extra = set(ds.data_vars) - {str(names.ltoa), 'data_quality'}
        assert not extra, (
            f'{self.sensor} {self.level}: unexpected grid fields {sorted(extra)}; '
            f'the v2 contract covers radiance and data_quality only')

        product = tree[spec['product_metadata']].to_dataset()
        cwvl_um = np.asarray(product['BandSpecification'].values).ravel()[1:]
        nbands = ds.sizes[str(names.bands)]
        assert len(cwvl_um) == nbands, (
            f'{self.sensor} {self.level}: BandSpecification carries '
            f'{len(cwvl_um)} thermal wavelengths for {nbands} data bands')
        ds[str(names.cwav)] = (str(names.bands), cwvl_um * 1e3)  # µm -> nm

        # The radiance fields are already calibrated: direct Planck inversion.
        valid = ~ds[str(names.ltoa)].isnull()
        cwvl = cwvl_um.reshape(nbands, 1, 1)
        ds[str(names.bt)] = _K2 / (cwvl * np.log(
            _K1 / (ds[str(names.ltoa)].where(valid) * cwvl ** 5) + 1))
        ds[str(names.bt)].attrs['unit'] = 'Kelvin'
        return ds


class Level2_Ecostress(_EcostressPlainBase):
    """ECOSTRESS level-2 land surface temperature and emissivity (LSTE).

    The input is the LSTE file; the paired L1B_GEO file must sit next to it
    and is auto-discovered by name substitution. Both PGE generations store
    the grid without per-pixel lat/lon, so the GEO pairing is required for
    every variant.

    Output: `LST` (primary, K) with its error, the per-band emissivity stack
    (`emis` / `emis_err`, cwav from the product's own BandSpecification), the
    wide-band emissivity (`EmisWB`), precipitable water vapor (`PWV`, cm) and
    the raw quality-control layer (`QC`; bit layout undocumented, so no flag
    decoding). The v002 variant additionally carries an in-grid cloud mask,
    exposed as `cloud` (0/1, NaN = invalid) and decoded by its dedicated flag
    reader.

    The v002 ShortName carries a scene suffix ('ECO_L2_LSTE.002'), so that
    variant is matched by prefix. The CLOUD products (ECO2CLD /
    ECOv002_L2_CLOUD) have a different output contract and are out of scope:
    they raise `UnsupportedProductType`.

    The GEO `height` and `line_start_time_j2000` fields, and the v002 grid's
    raw `water_mask`, are not part of the v2 contract and are dropped (the
    canonical `water` layer is derived from the GEO land_fraction, proven
    identical to the PGE mask).
    """

    level = 'L2'
    payload_var = 'LST'
    sample_products = ('ECOSTRESS_L2_V001', 'ECOSTRESS_L2_V002')
    required_vars = ('LST_err', 'emis', 'emis_err', 'EmisWB', 'PWV', 'QC',
                     'water', 'land_fraction')
    flags = 'eoread.readers.ecostress.FlagsReader_Level2'

    VARIANTS = {
        'ECO2LSTE': dict(
            detect={'field': 'ShortName', 'value': 'ECO2LSTE'},
            grid='SDS',
            product_metadata='L2 LSTE Metadata',
            geo_token=('L2_LSTE', 'L1B_GEO'),
            resolution=70,
        ),
        'ECO_L2_LSTE': dict(
            detect={'field': 'ShortName', 'value': 'ECO_L2_LSTE', 'prefix': True},
            grid='SDS',
            product_metadata='L2 LSTE Metadata',
            geo_token=('L2_LSTE', 'L1B_GEO'),
            resolution=70,
            flags='eoread.readers.ecostress.FlagsReader_Level2Cloud',
        ),
    }

    def payload(self, raw: xr.Dataset, tree: xr.DataTree, spec: dict) -> xr.Dataset:
        # Per-band emissivity fields merge into one band stack each; EmisWB
        # does not match the anchored patterns and passes through untouched.
        ds = merge(raw, dim=str(names.bands), varname='emis', pattern=r'^Emis(\d)$')
        ds = merge(ds, dim=str(names.bands), varname='emis_err',
                   pattern=r'^Emis(\d)_err$')

        # The v002 variant stores an in-grid cloud mask; v001 has none.
        if 'cloud_mask' in ds:
            ds = ds.rename({'cloud_mask': 'cloud'})

        # The v002 grid also carries the PGE's own water mask. The canonical
        # `water` layer is derived from the GEO land_fraction (proven
        # identical to this field), so the raw copy is dropped.
        if 'water_mask' in ds:
            ds = ds.drop_vars('water_mask')

        # Anything else in the grid is not part of the output contract.
        expected = {'LST', 'LST_err', 'emis', 'emis_err', 'EmisWB', 'PWV', 'QC'}
        if 'cloud' in ds:
            expected.add('cloud')
        extra = set(ds.data_vars) - expected
        assert not extra, (
            f'{self.sensor} {self.level}: unexpected grid fields {sorted(extra)}; '
            f'the v2 contract covers LST, emissivity, PWV and QC only')

        product = tree[spec['product_metadata']].to_dataset()
        cwvl_um = np.asarray(product['BandSpecification'].values).ravel()[1:]
        nbands = ds.sizes[str(names.bands)]
        assert len(cwvl_um) == nbands, (
            f'{self.sensor} {self.level}: BandSpecification carries '
            f'{len(cwvl_um)} thermal wavelengths for {nbands} data bands')
        ds[str(names.cwav)] = (str(names.bands), cwvl_um * 1e3)  # µm -> nm

        # The file stores the CF plural `units`; the registry wants `unit`.
        ds['LST'].attrs['unit'] = 'K'
        return ds


class FlagsReader_Level1B(FlagsReaderBase):
    """Generic flags of the ECOSTRESS v001 L1B radiance product.

    - L1_INVALID:   band-4 radiance is missing
    - L1_DEGRADED:  any data-quality layer is non-zero
    - LAND:         water mask is zero (derived from the GEO land_fraction)
    CLOUD has no source in this product and raises `ValueError`.
    """

    def requires(self) -> list[str]:
        return ['water', 'data_quality', str(names.ltoa)]

    def dims_like(self) -> str:
        return 'water'

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.L1_INVALID:
            band = ds[str(names.ltoa)].sel({str(names.bands): '4'})
            return band.isnull()
        if flag_name == GenericFlags.L1_DEGRADED:
            quality = ds['data_quality'].max(dim=str(names.bands))
            return quality != 0
        if flag_name == GenericFlags.LAND:
            return ds['water'] == 0
        raise ValueError(
            'CLOUD flag not available on ECOSTRESS L1B: the v001 product '
            'carries no cloud layer')

    def getflag_raw(self, ds: xr.Dataset, flag_name: str) -> xr.DataArray:
        raise NotImplementedError('ECOSTRESS flags are derived from data variables')


class FlagsReader_Level2(FlagsReaderBase):
    """Generic flags of the ECOSTRESS L2 LSTE product (v001 ECO2LSTE).

    - L1_INVALID:   LST is missing (the generic enum has no L2_INVALID)
    - LAND:         water mask is zero (derived from the GEO land_fraction)
    CLOUD has no source in this variant and raises `ValueError`.
    """

    def requires(self) -> list[str]:
        return ['LST', 'water']

    def dims_like(self) -> str:
        return 'LST'

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.L1_INVALID:
            return ds['LST'].isnull()
        if flag_name == GenericFlags.LAND:
            return ds['water'] == 0
        raise ValueError(
            'CLOUD flag not available on this ECOSTRESS L2 variant: the '
            'product carries no cloud layer')

    def getflag_raw(self, ds: xr.Dataset, flag_name: str) -> xr.DataArray:
        raise NotImplementedError('ECOSTRESS flags are derived from data variables')


class FlagsReader_Level2Cloud(FlagsReader_Level2):
    """v002 ECO_L2_LSTE adds the in-grid cloud mask (`cloud`, 0/1, NaN)."""

    def requires(self) -> list[str]:
        return ['LST', 'water', 'cloud']

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.CLOUD:
            # The mask is float32 with NaN invalids: only an explicit 1 is a
            # cloud (a `!= 0` test would count NaN pixels as cloudy).
            return ds['cloud'] == 1
        return super().getflag(ds, flag_name)
