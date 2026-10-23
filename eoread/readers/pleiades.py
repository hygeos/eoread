#!/usr/bin/env python3
"""
Pleiades and Pleiades-Neo DIMAP strip readers, level L1 (reflectance).

One class per sensor (REFACTORING.md, §4), sharing a sensor base:

    >>> from eoread.readers.pleiades import Level1_PLEIADES        # doctest: +SKIP
    >>> ds = Level1_PLEIADES('Polygon1_SO24012539-2-01_DS_...')    # doctest: +SKIP

Differences from the legacy v1 readers (documented deviations, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the strip (canonical
  orientation). Legacy outputs were north-up.
- Nodata pixels (DN == 0 per the DIMAP Special_Value table) become NaN; the
  legacy rioxarray open kept them as zero data (masked=False default).
- The payload, tie angles and lat/lon are float32 end to end; the legacy
  reader mixed float64 intermediates (rtoa was computed in float64 from
  float32 calibration values).
- `datetime` uses the ISO-8601 'T' separator (legacy used a space between
  date and time).
"""

import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from xarray import open_dataarray

from core import log
from core.geo.naming import names
from core.table import read_xml

from eoread.reader import ProductError, Reader
from eoread.tools import (_blockwise_interp, filter_metadata,
                          install_exit_cleanup, _num, xml_value)


def read_srf_pleiades(
        sensor_name: str,
        srf_file: Path | None = None,
        panchromatic: bool = False,
) -> xr.Dataset:
    """
    Read the Pleiades / Pleiades-Neo spectral response functions.

    One variable per band (plus `PAN` when `panchromatic=True`) on a shared
    `wavelength` coordinate in nm. The Excel file is downloaded from the eoread
    release assets and cached under DIR_STATIC on first use.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor (e.g. 'PHR1B', 'PNEO4'), stored as the `sensor`
        attribute of every returned band.
    srf_file : Path, optional
        Path to an Excel file containing the SRF data. Downloaded when None.
    panchromatic : bool, optional
        Whether to keep the panchromatic band in the result.
    """
    # Local imports: keep this module importable without the static-data and
    # network stack (the smoke gate imports every reader module).
    import pandas as pd
    from core import env
    from core.files import mdir
    from core.network.download import download_url

    if srf_file is None:
        url = ('https://github.com/hygeos/eoread/releases/download/root/'
               'SpectralValues_Normalized-PHR1A_PHR1B_SPOT6_SPOT7_PNEO3-4.xlsx')
        srf_file = download_url(url, mdir(env.getdir('DIR_STATIC') / 'srf' / 'pleiades'))

    # The column is labelled mm but holds um values; x1000 gives nm (legacy).
    df = pd.read_excel(srf_file, sheet_name=0)
    assert 'Lambda(mm)' in df.columns

    ds = xr.Dataset()
    for band in (col for col in df.columns if col != 'Lambda(mm)'):
        ds[band] = xr.DataArray(df[band].values, dims=['wavelength'], name=band,
                                attrs={'sensor': sensor_name})
    ds = ds.assign_coords(wavelength=(df['Lambda(mm)'].values * 1000))
    ds.wavelength.attrs.update(units='nm')

    if not panchromatic:
        return ds[[x for x in ds if x != 'PAN']]
    return ds


def _angle_block(block, block_info, pts, vals):
    """Interpolate a 3-point Use_Area profile onto one output block.

    `pts`/`vals` are the row positions and values of the Top / Center /
    Bottom Use_Area entries; rows outside the profile are clamped, matching
    the legacy interpolation with bounds='clip'. The profile is constant
    along x (the entries are center-line values), so each row value is
    replicated across the block width.
    """
    r0 = block_info[0]['array-location'][0][0]
    rows = np.arange(r0, r0 + block.shape[0], dtype='float64')
    prof = np.interp(rows, pts, vals)
    return np.repeat(prof[:, None], block.shape[1], axis=1).astype('float32')


class _DimapBase(Reader):
    """Shared DIMAP strip implementation for the Pleiades family.

    Subclasses provide `_open_payload` (band rasters + labels) and
    `_band_table` (central wavelengths, gains, biases); footprint lat/lon,
    Use_Area angles, calibration and canonical attributes are common to both
    sensors.
    """

    sensor = None
    level = 'L1'
    chunks_default = 1024
    payload_var = str(names.rtoa)
    payload_unit = None
    required_vars = (str(names.sza), str(names.saa),
                     str(names.vza), str(names.vaa))
    sample_products = ()
    srf_getter = 'eoread.readers.pleiades.read_srf_pleiades'

    # ------------------------------------------------------------- discovery

    @staticmethod
    def _find_strip(product: Path) -> tuple[Path, Path]:
        """Descend single-child directories to the DIMAP strip directory.

        Returns (strip_dir, dim_path). A product that is already a strip
        directory is accepted as-is; when several subdirectories remain, the
        unique MS-named one is selected.
        """
        prod = product
        subdirs = sorted(d for d in prod.iterdir() if d.is_dir())
        while len(subdirs) == 1:
            prod = subdirs[0]
            subdirs = sorted(d for d in prod.iterdir() if d.is_dir())
        if len(subdirs) >= 2:
            ms = [d for d in subdirs if 'MS' in d.name]
            assert len(ms) == 1, (
                f'expected a single MS strip directory under {prod}, '
                f'got {[d.name for d in subdirs]}')
            prod = ms[0]
        dims = sorted(prod.glob('DIM*.XML'))
        assert len(dims) == 1, (
            f'expected exactly one DIM*.XML in {prod}, got {len(dims)}')
        return prod, dims[0]

    # ------------------------------------------------------------ metadata

    @staticmethod
    def _strip_source(dim: dict) -> dict:
        """Return the Strip_Source entry of Source_Identification.

        read_xml returns a singleton child as a dict and repeated children as
        a list; both shapes are normalized here.
        """
        ident = dim['Dataset_Sources']['Source_Identification']
        entries = [ident] if isinstance(ident, dict) else list(ident)
        for entry in entries:
            if entry.get('SOURCE_TYPE') == 'Strip_Source':
                assert 'Strip_Source' in entry, \
                    f'Strip_Source entry without a Strip_Source node: {entry}'
                return entry['Strip_Source']
        raise ProductError(f'no Strip_Source entry in Source_Identification of {dim}')

    def _latlon(self, dim: dict, ny: int, nx: int) -> tuple:
        """Bilinear lat/lon grid from the four footprint corners.

        File space (row 0 = north), computed lazily with one task per output
        block; bit-for-bit identical to a full-grid core.interpolate.interp
        evaluation on the same corner values.
        """
        verts = dim['Dataset_Content']['Dataset_Extent']['Vertex']
        assert len(verts) == 4, f'expected 4 footprint vertices, got {len(verts)}'
        # Document order is clockwise [TL, TR, BR, BL]; sorting by (row, col)
        # yields the raster order [TL, TR, BL, BR] used by the legacy reader.
        tl, tr, bl, br = sorted(verts, key=lambda v: (int(v['ROW']), int(v['COL'])))
        rows = np.array([int(tl['ROW']) - 1, int(bl['ROW']) - 1], dtype='float64')
        cols = np.array([int(tl['COL']) - 1, int(tr['COL']) - 1], dtype='float64')
        assert rows[0] < rows[-1] and cols[0] < cols[-1], 'footprint corners are not ordered'

        lat_g = np.array([[float(tl['LAT']), float(tr['LAT'])],
                          [float(bl['LAT']), float(br['LAT'])]], dtype='float32')
        lon_g = np.array([[float(tl['LON']), float(tr['LON'])],
                          [float(bl['LON']), float(br['LON'])]], dtype='float32')

        # Targets clipped to the corner span: legacy interpolated with
        # bounds='clip' and the corners can fall slightly outside the grid.
        ty = np.clip(np.arange(ny), rows[0], rows[-1]).astype('float64')
        tx = np.clip(np.arange(nx), cols[0], cols[-1]).astype('float64')
        chunks = (self.chunks[str(names.rows)], self.chunks[str(names.columns)])

        lat = _blockwise_interp(lat_g, (rows, cols), (ty, tx), chunks).astype('float32')
        lon = _blockwise_interp(lon_g, (rows, cols), (ty, tx), chunks).astype('float32')
        return lat, lon

    def _angles(self, dim: dict, ny: int, nx: int) -> tuple:
        """Use_Area 3-point profiles interpolated onto the full grid.

        The DIMAP carries Top / Center / Bottom center-line values; each angle
        is linearly interpolated along y (clamped at the ends, as legacy) and
        replicated along x — lazily, one task per output block.
        """
        vals = dim['Geometric_Data']['Use_Area']['Located_Geometric_Values']
        assert len(vals) == 3, f'expected 3 Use_Area entries, got {len(vals)}'
        pts = np.array([0.0, (ny - 1) / 2.0, ny - 1.0], dtype='float64')
        chunks = (self.chunks[str(names.rows)], self.chunks[str(names.columns)])
        dummy = da.zeros((ny, nx), chunks=chunks)

        def profile(parent: str, key: str, transform):
            values = np.array([transform(float(xml_value(e[parent][key])))
                               for e in vals], dtype='float64')
            return da.map_blocks(_angle_block, dummy, pts=pts, vals=values,
                                 dtype='float32')

        sza = profile('Solar_Incidences', 'SUN_ELEVATION', lambda v: 90.0 - v)
        saa = profile('Solar_Incidences', 'SUN_AZIMUTH', float)
        vza = profile('Acquisition_Angles', 'VIEWING_ANGLE', float)
        vaa = profile('Acquisition_Angles', 'AZIMUTH_ANGLE', float)
        return sza, saa, vza, vaa

    def _resolution(self, dim: dict) -> float:
        """Mean across/along ground sample distance of the center Use_Area entry."""
        gsd = (dim['Geometric_Data']['Use_Area']
               ['Located_Geometric_Values'][1]['Ground_Sample_Distance'])
        return (float(xml_value(gsd['GSD_ACROSS_TRACK']))
                + float(xml_value(gsd['GSD_ALONG_TRACK']))) / 2.0

    # ------------------------------------------------------------ payload

    def _lazy_open_jp2(self, path: Path) -> xr.DataArray:
        """Open a JP2 lazily; drop the file's bogus J2W coordinates."""
        chunks = {'y': self.chunks[str(names.rows)],
                  'x': self.chunks[str(names.columns)]}
        with warnings.catch_warnings():
            # xarray's chunk normalizer warns when our chunk boundaries fall
            # inside the file's stored tiles (one-shot windowed read; the
            # user-facing chunk spec takes precedence, so the hint is noise).
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message='The specified chunks separate the stored chunks')
            arr = open_dataarray(str(path), engine='rasterio',
                                 chunks=chunks).rename({'band': str(names.bands)})
        install_exit_cleanup()
        for coord in ('x', 'y', 'spatial_ref'):
            if coord in arr.coords:
                arr = arr.drop_vars(coord)
        # Cast before masking: .where on an integer array would promote to
        # float64 and break the all-float32 chain.
        return arr.astype('float32')

    @staticmethod
    def _nodata_values(display: dict) -> list[float]:
        """DN values flagged NODATA in a Raster_Display node."""
        specials = display['Special_Value']
        if isinstance(specials, dict):
            specials = [specials]
        values = []
        for sv in specials:
            if str(xml_value(sv.get('SPECIAL_VALUE_TEXT', ''))) == 'NODATA':
                values.append(_num(sv.get('SPECIAL_VALUE_COUNT'), default=0.0))
        return values

    @staticmethod
    def _mask_nodata(dn: xr.DataArray, values) -> xr.DataArray:
        for value in values:
            dn = dn.where(dn != value)
        return dn

    # --------------------------------------------------------------- read

    def read(self, path: Path) -> xr.Dataset:
        product = Path(path).resolve()
        assert product.is_dir(), f'{self.sensor} product not found: {product}'

        if self.verbose:
            log.debug(f'Reading {self.sensor} L1 strip {product.name}')

        strip, dim_path = self._find_strip(product)
        dim = read_xml(str(dim_path))
        src = self._strip_source(dim)

        dn, bands = self._open_payload(strip, dim)
        ny, nx = int(dn.sizes[str(names.rows)]), int(dn.sizes[str(names.columns)])
        cwav, gains, bias = self._band_table(dim, bands)

        # Geolocation and angles in file (north-up) space; the canonical
        # south-up flip happens once at the end.
        lat, lon = self._latlon(dim, ny, nx)
        sza, saa, vza, vaa = self._angles(dim, ny, nx)

        g = xr.DataArray(gains, dims=(str(names.bands),))
        b = xr.DataArray(bias, dims=(str(names.bands),))
        rtoa = (dn / g + b) / np.cos(np.radians(sza))

        # Canonical band order: stable ascending sort by central wavelength.
        order = list(np.argsort(cwav, kind='stable'))
        rtoa = rtoa.isel(bands=order)
        bands = [bands[i] for i in order]
        cwav = cwav[order]

        ds = xr.Dataset({self.payload_var: rtoa})
        ds = ds.assign_coords({str(names.rows): np.arange(ny),
                               str(names.columns): np.arange(nx)})
        for var, data in ((str(names.lat), lat), (str(names.lon), lon),
                          (str(names.sza), sza), (str(names.saa), saa),
                          (str(names.vza), vza), (str(names.vaa), vaa)):
            ds[var] = ((str(names.rows), str(names.columns)), data)

        # Canonical orientation: row 0 = southern edge (single vertical flip)
        ds = ds.isel(y=slice(None, None, -1))
        ds = ds.assign_coords({
            str(names.rows): np.arange(ny),
            str(names.columns): np.arange(nx),
        })

        # Spectral coordinates
        ds[str(names.cwav)] = (str(names.bands), cwav.astype('float64'))
        ds[str(names.cwav)].attrs['unit'] = 'nm'
        ds[str(names.bgroup)] = (str(names.bands), ['bands_vnir'] * len(bands))
        ds = ds.set_coords(str(names.bgroup))

        # Curated attributes (legacy parity + schema requirements)
        if self.metadata_template is None:
            metadata = dim
        else:
            metadata = filter_metadata(dim, self.metadata_template)
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): f'{src["MISSION"]}-{src["MISSION_INDEX"]}',
            str(names.product_name): str(xml_value(
                dim['Dataset_Identification']['DATASET_NAME'])),
            str(names.input_directory): str(product.parent),
            str(names.datetime): (f'{src["IMAGING_DATE"]}T{src["IMAGING_TIME"]}'
                                  ).removesuffix('Z'),
            str(names.resolution): self._resolution(dim),
            str(names.shortname): str(src['BAND_MODE']),
            str(names.crs): 'EPSG:4326',
            '_srf_getter': self.srf_getter,
            '_srf_getter_arg': f'{src["MISSION"]}{src["MISSION_INDEX"]}',
            'metadata': metadata,
        })
        ds[self.payload_var].attrs['unit'] = self.payload_unit

        return ds

    # ---------------------------------------------------- subclass hooks

    def _open_payload(self, strip: Path, dim: dict) -> tuple[xr.DataArray, list]:
        """Return (DN DataArray (bands, y, x) float32, band labels)."""
        raise NotImplementedError

    def _band_table(self, dim: dict, bands: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cwav float64, gains float32, bias float32) in `bands` order."""
        raise NotImplementedError


class Level1_PLEIADES(_DimapBase):
    """Pleiades multi-spectral L1 reflectance strips (PX mode)."""

    sensor = 'PLEIADES'
    sample_products = ('PLEIADES_MS',)

    def _open_payload(self, strip: Path, dim: dict) -> tuple[xr.DataArray, list]:
        nbands = int(dim['Raster_Data']['Raster_Dimensions']['NBANDS'])
        assert nbands == 4, f'expected a 4-band MS product, got {nbands} bands'
        jp2 = sorted(strip.glob('*R1C1.JP2'))
        assert len(jp2) == 1, (
            f'expected exactly one *R1C1.JP2 in {strip}, got {len(jp2)}')
        # File order B0..B3 is the sensor band order.
        bands = ['BLUE', 'GREEN', 'RED', 'NIR']
        dn = self._lazy_open_jp2(jp2[0])
        display = dim['Raster_Data']['Raster_Display']
        dn = self._mask_nodata(dn, self._nodata_values(display))
        return dn.assign_coords({str(names.bands): bands}), bands

    def _band_table(self, dim: dict, bands: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bml = (dim['Radiometric_Data']['Radiometric_Calibration']
               ['Instrument_Calibration']['Band_Measurement_List'])
        spec = bml['Band_Spectral_Range']
        refl = bml['Band_Reflectance']
        if isinstance(spec, dict):
            spec = [spec]
        if isinstance(refl, dict):
            refl = [refl]
        assert len(spec) == len(bands) and len(refl) == len(bands), \
            f'band table has {len(spec)}/{len(refl)} entries for {len(bands)} bands'
        # Band_Spectral_Range MIN/MAX are in um; central wavelength in nm.
        # Scale to nm before averaging so the result is exact for the usual
        # 4-decimal um bounds (e.g. (0.5475, 0.5625) -> 555.0, not 554.999...).
        cwav = np.array([0.5 * (float(_num(s['MIN'])) * 1000 + float(_num(s['MAX'])) * 1000)
                         for s in spec], dtype='float64')
        gains = np.array([float(_num(s['GAIN'])) for s in refl], dtype='float32')
        bias = np.array([_num(s['BIAS'], default=0.0) for s in refl], dtype='float32')
        return cwav, gains, bias
