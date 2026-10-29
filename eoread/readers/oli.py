#!/usr/bin/env python3
"""
Landsat OLI (Operational Land Imager) reader, level 1TP (Landsat-8 / Landsat-9).

One class per level (REFACTORING.md, §4), sharing a sensor base:

    >>> from eoread.readers.oli import Level1_OLI        # doctest: +SKIP
    >>> ds = Level1_OLI('LC08_L1TP_...')    # doctest: +SKIP

Differences from the legacy v1 reader (documented clean breaks, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the scene (canonical
  orientation). Legacy outputs were north-up.
- The default output matches the legacy layout: the 10 bands at 30 m stay on
  their native grid and the panchromatic band B8 is carried separately as a
  `Panchromatic` variable (calibrated radiance) on its own 15 m
  'rows_pan'/'columns_pan' dims, with no resampling at all
  (`resolution=None`). Passing `resolution=15` or `30` instead puts every
  band on a single uniform grid and resamples the others onto it lazily.
- `Rtoa` is the declared primary payload: REFLECTANCE_MULT * DN +
  REFLECTANCE_ADD, as defined by USGS C2. The legacy reader additionally
  divided by cos(solar zenith angle), double-applying the solar correction
  that the published reflectance factors already include.
- `Ltoa` (W.m-2.sr-1.um-1) and `BT` (K, B10/B11 only) are declared extra
  payloads with unit attributes; in legacy they were plain variables without
  units.
- Central wavelengths come from a per-mission table keyed by band label
  (legacy assigned one bare positional list to the sorted bands dim); the pan
  band, excluded from the legacy bands dim, is included here.
- `QA_PIXEL` is kept raw uint16 on the pixel grid and decoded on demand by
  the reader's `getflag` into a boolean L1_INVALID; the legacy reader exposed the
  raw unbooleaned bitmask and also read QA_RADSAT, which has no consumer in v2.
- The dataset carries every canonical attribute, in particular
  'input_directory'; its absence made the legacy output fail the generic
  harness.
"""

import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import rasterio
import xarray as xr
from xarray import open_dataarray

from core import log
from core.geo.naming import names
from core.table import read_xml

from eoread.flags import GenericFlags
from eoread.reader import ProductError, Reader
from eoread.tools import filter_metadata, install_exit_cleanup, spatial_resample
from eoread.transforms.geometry import latlon_utm


#: Uniform-grid target resolutions (m) for `resolution=15`/`30`: 15 m is the
#: panchromatic native grid, 30 m the multispectral one. The default
#: `resolution=None` keeps every band on its own native grid (no resampling).
RESOLUTIONS = (15, 30)

#: USGS Landsat Data Users Handbook (same reference as the legacy reader).
_USER_GUIDE = 'https://greenpolicy360.net/images/Landsat8DataUsersHandbook.pdf'

#: Per-mission central wavelengths (nm), keyed by band label.
#:
#: L8 values are the legacy reader's table; pan B8 is new in v2 (the legacy
#: bands dim excluded it). L9 OLI-2 values per the eoPortal sensor
#: specification (cirrus 1375 nm); the thermal bands keep the TIRS values of
#: the USGS handbook. eotools is not installed in the dev environment, so the
#: table is hardcoded here instead of being fetched at read time.
_CWVL = {
    'L8': {'B1': 442.96, 'B2': 482.04, 'B3': 561.41, 'B4': 654.59,
           'B5': 864.67, 'B6': 1608.86, 'B7': 2200.73, 'B8': 590.0,
           'B9': 1373.43, 'B10': 10895.0, 'B11': 12050.0},
    'L9': {'B1': 443.0, 'B2': 482.0, 'B3': 562.0, 'B4': 655.0,
           'B5': 865.0, 'B6': 1610.0, 'B7': 2200.0, 'B8': 590.0,
           'B9': 1375.0, 'B10': 10895.0, 'B11': 12050.0},
}

#: All L1TP bands in band-number order, with their native resolution (m).
_BANDS = (('B1', 30), ('B2', 30), ('B3', 30), ('B4', 30), ('B5', 30),
          ('B6', 30), ('B7', 30), ('B8', 15), ('B9', 30),
          ('B10', 30), ('B11', 30))


def _only_file(dirname: Path, pattern: str) -> Path:
    """The single file matching `pattern` in `dirname`; loud otherwise."""
    matches = list(Path(dirname).glob(pattern))
    if len(matches) != 1:
        raise ProductError(
            f'expected exactly one {pattern} in {dirname}, found {len(matches)}')
    return matches[0]


def _all_nan(dn):
    """All-NaN float32 twin of `dn`, sharing its dims, coords and chunks."""
    return dn.copy(data=da.full(dn.shape, np.nan, dtype='float32', chunks=dn.chunks))


def _open(path: Path, chunks: dict, mask_nodata: bool = True) -> xr.DataArray:
    """Open one granule file lazily as a 2D (y, x) DataArray.

    ``mask_and_scale=False`` keeps the native integer dtype: the reader applies
    its own calibration from the MTL factors, so no file-side scaling is
    wanted.  The file's GDAL nodata value is nevertheless honoured: the band
    rasters carry ``GDAL_NODATA=0`` and the rotated-scene corners (~30% of the
    raster on a typical L1TP granule) are exactly those fill pixels.  Masking
    them to NaN before calibration is what prevents the fill value from being
    turned into a plausible-looking negative reflectance.  Set
    ``mask_nodata=False`` for integer bit-fields such as QA_PIXEL, where the
    bit pattern must be preserved verbatim.

    The chunking warning is expected and harmless: the TIFs are stored in
    256 px tiles while we read the canonical (larger) output windows.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The specified chunks separate')
        arr = open_dataarray(path, engine='rasterio', mask_and_scale=False,
                             chunks=chunks).squeeze()
    # squeeze drops the size-1 band dim but leaves its scalar coordinate (and
    # spatial_ref) behind; they are file artifacts with no canonical meaning.
    for coord in ('band', 'spatial_ref'):
        if coord in arr.coords:
            arr = arr.drop_vars(coord)

    if mask_nodata:
        nodata = arr.rio.nodata
        if nodata is not None:
            arr = arr.where(arr != nodata)

    return arr


class _OliBase(Reader):
    """Shared implementation of the OLI level readers."""

    sensor = 'OLI'
    chunks_default = 500
    flags = True
    required_vars = (str(names.ltoa), str(names.bt), 'QA_PIXEL',
                     str(names.sza), str(names.saa), str(names.vza), str(names.vaa))

    VARIANTS = {
        'L8': dict(
            detect=dict(field=('IMAGE_ATTRIBUTES', 'SPACECRAFT_ID'), value='LANDSAT_8'),
            platform='Landsat 8',
            cwvl=_CWVL['L8'],
            srf_arg='landsat_8_oli'),
        'L9': dict(
            detect=dict(field=('IMAGE_ATTRIBUTES', 'SPACECRAFT_ID'), value='LANDSAT_9'),
            platform='Landsat 9',
            cwvl=_CWVL['L9'],
            srf_arg='landsat_9_oli'),
    }

    # Per-level hooks, set by the subclasses ---------------------------------
    level = None
    payload_var = None

    def __init__(self, path, chunks=None, metadata_template=None, verbose=False, resolution=None):
        if resolution is not None and resolution not in RESOLUTIONS:
            raise ValueError(
                f'OLI {self.level}: unsupported resolution {resolution!r}, '
                f'expected one of (None, *{RESOLUTIONS})')
        self.resolution = resolution
        super().__init__(path, chunks=chunks, metadata_template=metadata_template, verbose=verbose)

    # ------------------------------------------------------------------ read

    def read(self, path: Path) -> xr.Dataset:
        dirname = Path(path).resolve()
        if not dirname.is_dir():
            raise ProductError(f'OLI {self.level}: {dirname} is not a directory')

        mtl = read_xml(_only_file(dirname, 'LC*_MTL.xml'))
        spec = self.VARIANTS[self.detect_variant(mtl)]

        if self.verbose:
            log.debug(f'Reading OLI {self.level} product {dirname.name}')

        # Grids ----------------------------------------------------------------
        # resolution=None (default): the v1-compatible native layout — the 30 m
        # bands keep their own grid and B8 pan is carried separately at 15 m;
        # nothing is resampled. A uniform target resolution picks one reference
        # band (B8 pan at 15 m, B4 red at 30 m) whose shape and georeferencing
        # define the single output grid for every band.
        rows, cols = str(names.rows), str(names.columns)
        bands = str(names.bands)
        ref_band = 'B8' if (self.resolution == 15) else 'B4'
        with rasterio.open(_only_file(dirname, f'LC*_{ref_band}.TIF')) as src:
            transform = src.transform
            epsg = src.crs.to_epsg()
            ny, nx = int(src.height), int(src.width)
        if epsg is None:
            raise ProductError(
                f'OLI {self.level}: unexpected non-EPSG CRS {src.crs!s} in {ref_band}.TIF')
        crs = f'EPSG:{epsg}'

        # Panchromatic grid (B8, 15 m); the main lat/lon and angles stay on
        # the 30 m reference grid, as in legacy.
        rows_pan = cols_pan = None
        ny_p = nx_p = 0
        if self.resolution is None:
            rows_pan, cols_pan = rows + '_pan', cols + '_pan'
            b8 = list(dirname.glob('LC*_B8.TIF'))
            if len(b8) == 1:
                with rasterio.open(b8[0]) as src:
                    ny_p, nx_p = int(src.height), int(src.width)

        resc = mtl['LEVEL1_RADIOMETRIC_RESCALING']
        therm = mtl['LEVEL1_THERMAL_CONSTANTS']

        shape = {rows: ny, cols: nx}

        # Radiometry ----------------------------------------------------------
        # One lazy DN read per band, shared by the Ltoa/Rtoa/BT graphs. USGS
        # C2 calibration factors from the MTL; the reflectance factors (B1-B9)
        # already fold in the solar geometry, so no cos(sza) division. In the
        # native layout (resolution=None) each band stays on its own grid and
        # is not resampled; B8 becomes the separate `Panchromatic` variable.
        ltoa_parts, rtoa_parts, bt_parts = [], [], []
        kept_bands = []
        pan_ltoa = None
        for band, native_res in _BANDS:
            num = int(band[1:])
            matches = list(dirname.glob(f'LC*_{band}.TIF'))
            if len(matches) != 1:
                log.info(f'OLI {self.level}: no data file for band {band} in {dirname}, skipping')
                continue
            dn = _open(matches[0], self.chunks)
            dn = dn.rename(x=cols, y=rows)
            if self.resolution is None and native_res == 15:
                # Native layout: the pan band lives on its own grid.
                dn = dn.rename({rows: rows_pan, cols: cols_pan})
                dn = dn.assign_coords(
                    {rows_pan: np.arange(dn.sizes[rows_pan]),
                     cols_pan: np.arange(dn.sizes[cols_pan])})
            else:
                dn = dn.assign_coords({rows: np.arange(dn.sizes[rows]), cols: np.arange(dn.sizes[cols])})

            ltoa = (dn * float(resc[f'RADIANCE_MULT_BAND_{num}'])
                    + float(resc[f'RADIANCE_ADD_BAND_{num}'])).astype('float32')

            if f'REFLECTANCE_MULT_BAND_{num}' in resc:
                rtoa = (dn * float(resc[f'REFLECTANCE_MULT_BAND_{num}'])
                        + float(resc[f'REFLECTANCE_ADD_BAND_{num}'])).astype('float32')
            else:
                rtoa = _all_nan(dn)

            if f'K1_CONSTANT_BAND_{num}' in therm:
                k1 = float(therm[f'K1_CONSTANT_BAND_{num}'])
                k2 = float(therm[f'K2_CONSTANT_BAND_{num}'])
                # Planck inversion, guarded against non-positive radiance.
                safe = xr.where(ltoa > 0, ltoa, 1.0)
                bt = xr.where(ltoa > 0, k2 / np.log1p(k1 / safe), np.nan)
            else:
                bt = _all_nan(dn)

            if self.resolution is None and native_res == 15:
                # Legacy carried the pan radiance as a separate 2D variable.
                pan_ltoa = ltoa
            else:
                # Resample onto the target grid and pin the payload dtype; the
                # interpolation branch returns swapped dim labels (see
                # spatial_resample).
                resampled = []
                for arr in (ltoa, rtoa, bt):
                    arr = spatial_resample(arr, shape, self.chunks).transpose(rows, cols)
                    if arr.dtype != np.float32:
                        arr = arr.astype('float32')
                    resampled.append(arr)
                ltoa, rtoa, bt = resampled

                ltoa_parts.append(ltoa)
                rtoa_parts.append(rtoa)
                bt_parts.append(bt)
                kept_bands.append((band, native_res))

        if not kept_bands and pan_ltoa is None:
            raise ProductError(f'OLI {self.level}: no band data file found in {dirname}')

        ds = xr.Dataset()
        if kept_bands:
            for var, parts in ((str(names.ltoa), ltoa_parts),
                               (str(names.rtoa), rtoa_parts),
                               (str(names.bt), bt_parts)):
                ds[var] = xr.DataArray(da.stack([a.data for a in parts]), dims=(bands, rows, cols))
        if pan_ltoa is not None:
            ds['Panchromatic'] = pan_ltoa

        # Viewing/solar angles: int16 rasters in 0.01 degrees on the 30 m grid;
        # resampled onto the target grid only in uniform-grid mode (smooth
        # fields, linear is fine). USGS stores the azimuths signed (negative
        # east of the meridian); the canonical schema keeps the unsigned [0, 180]
        # angle, so fold them on the native grid before resampling.
        for tag, name, unsigned in (('SZA', str(names.sza), False),
                                    ('SAA', str(names.saa), True),
                                    ('VZA', str(names.vza), False),
                                    ('VAA', str(names.vaa), True)):
            arr = _open(_only_file(dirname, f'LC*_{tag}.TIF'), self.chunks)
            arr = arr.rename(x=cols, y=rows)
            arr = arr.assign_coords({rows: np.arange(arr.sizes[rows]), cols: np.arange(arr.sizes[cols])})
            arr = (arr / 100.0).astype('float32')
            if unsigned:
                arr = np.abs(arr)
            if self.resolution is not None:
                arr = spatial_resample(arr, shape, self.chunks).transpose(rows, cols)
                if arr.dtype != np.float32:
                    arr = arr.astype('float32')
            ds[name] = arr

        # QA_PIXEL kept raw uint16 on the pixel grid (decoded on demand by
        # the reader's getflag); nearest-neighbour resampling in uniform-grid
        # mode so the bit patterns survive.
        qa = _open(_only_file(dirname, 'LC*_QA_PIXEL.TIF'), self.chunks,
                   mask_nodata=False)
        qa = qa.rename(x=cols, y=rows)
        qa = qa.assign_coords({rows: np.arange(qa.sizes[rows]), cols: np.arange(qa.sizes[cols])})
        if self.resolution is not None:
            qa = spatial_resample(qa, shape, self.chunks, 'repeat')
        ds['QA_PIXEL'] = qa.transpose(rows, cols)

        # Pixel geolocation from the reference file's affine transform; lazy.
        # `e` is passed signed: for these north-up files it is negative, so the
        # northing vector decreases with row index (row 0 = top of the scene).
        ds = ds.assign(latlon_utm(
            crs, float(transform.c), float(transform.f),
            float(transform.a), float(transform.e), nx, ny, self.chunks))

        # Canonical orientation: row 0 is the southern edge (the granule files
        # are north-up); the pan grid flips with it. Both grids flip in one
        # Dataset.isel: assigning a pre-flipped Panchromatic back into the
        # dataset would re-align it against the ascending y_pan coordinate and
        # silently undo the flip.
        flip = {rows: slice(None, None, -1)}
        if pan_ltoa is not None:
            flip[rows_pan] = slice(None, None, -1)
        ds = ds.isel(flip)
        coords = {rows: np.arange(ny), cols: np.arange(nx)}
        if pan_ltoa is not None:
            coords.update({rows_pan: np.arange(ny_p), cols_pan: np.arange(nx_p)})
        ds = ds.assign_coords(coords)

        # Spectral coordinates: band labels, central wavelengths and the
        # resolution-based band groups (as for MSI/MODIS).
        ds = ds.assign_coords({
            bands: [b for b, _ in kept_bands],
            str(names.cwav): (bands, np.array([spec['cwvl'][b] for b, _ in kept_bands],
                                              dtype='float64')),
            str(names.bgroup): (bands, [f'bands_{r}m' for _, r in kept_bands]),
        })
        ds[str(names.cwav)].attrs['unit'] = 'nm'

        # Attributes ----------------------------------------------------------
        ia = mtl['IMAGE_ATTRIBUTES']
        pc = mtl['PRODUCT_CONTENTS']
        metadata = mtl if self.metadata_template is None \
            else filter_metadata(mtl, self.metadata_template)
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): spec['platform'],
            str(names.product_name): str(pc['LANDSAT_PRODUCT_ID']),
            str(names.input_directory): str(dirname),
            str(names.datetime): f"{ia['DATE_ACQUIRED']}T{ia['SCENE_CENTER_TIME']}",
            # Native layout: the representative (multispectral) resolution, as
            # in legacy; uniform-grid mode carries the requested one.
            str(names.resolution): 30 if self.resolution is None else self.resolution,
            str(names.shortname): str(pc['PROCESSING_LEVEL']),
            str(names.crs): crs,
            'user_guide': _USER_GUIDE,
            '_srf_getter': 'eotools.srf.get_SRF_eumetsat',
            '_srf_getter_arg': spec['srf_arg'],
            'metadata': metadata,
        })

        install_exit_cleanup()
        return self.finalize(ds.unify_chunks())

    # ----------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return ['QA_PIXEL']

    @classmethod
    def flag_dims_like(cls) -> str:
        return 'QA_PIXEL'

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """`QA_PIXEL` is kept raw; L1_INVALID = any USGS C2 quality bit set.

        Bit 0 is reserved; fill | shadow | snow/ice | water | cloud (medium
        confidence) | cloud (high confidence). QA_RADSAT is not read in v2.
        """
        if flag_name == GenericFlags.L1_INVALID:
            return (ds['QA_PIXEL'] & 0b1111110) != 0
        raise ValueError(f'Unsupported OLI flag: {flag_name!r}')


class Level1_OLI(_OliBase):
    """Landsat-8/9 OLI-TIRS level 1TP; primary payload `Rtoa` (USGS C2 TOA reflectance)."""

    level = 'L1TP'
    payload_var = str(names.rtoa)
    sample_products = ('OLI_L1TP_L8', 'OLI_L1TP_L9')
