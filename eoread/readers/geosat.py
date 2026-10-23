#!/usr/bin/env python3
"""
GEOSAT-1 (DE-1) and GEOSAT-2 (DE-2) DIMAP readers, level L1 (radiance).

One class for both satellites; the variant is detected from the DIMAP
Scene_Source mission name (REFACTORING.md, §4):

    >>> from eoread.readers.geosat import Level1_GEOSAT            # doctest: +SKIP
    >>> ds = Level1_GEOSAT('DE00bb31s_L1R')                        # doctest: +SKIP

Differences from the legacy v1 readers (documented deviations, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the scene (canonical
  orientation). Legacy outputs were north-up.
- GEOSAT-2 nodata pixels (DN == 0) become NaN; the legacy rioxarray open kept
  them as zero data (masked=False default).
- DE-2 `datetime` is the full START_TIME; the legacy reader produced a
  date-only value (its IMAGING_TIME lookup found nothing for DE-2).
- `cwav` is float64 (legacy int32) and `crs` / `shortname` attributes are set
  from the product (legacy did not).
"""

import warnings
from pathlib import Path
from typing import Literal

import dask.array as da
import numpy as np
import rasterio
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from xarray import open_dataarray

from core import log
from core.geo.naming import names
from core.table import read_xml

from eoread.reader import Reader
from eoread.tools import (filter_metadata, install_exit_cleanup, _num, xml_value)
from eoread.transforms.geometry import latlon_utm


def read_geosat_srf(
        sensor: Literal['DE-1', 'DE-2'],
        file_path: str | Path | None = None,
        panchromatic: bool = False,
) -> xr.Dataset:
    """
    Read GEOSAT Spectral Response Function (SRF) data from Excel file.

    Parameters
    ----------
    sensor : {'DE-1', 'DE-2'}
        Sensor identifier.
    file_path : Path or str, optional
        Path to the Excel file containing SRF data. Downloaded from the eoread
        release assets and cached under DIR_STATIC when None.
    panchromatic : bool, optional
        Whether to keep a panchromatic band in the result (none present in the
        GEOSAT SRF tables; kept for signature parity).

    Returns
    -------
    xr.Dataset
        Dataset containing one variable per band on its own `wav_<band>`
        wavelength coordinate (nm), sorted by increasing weighted-average
        wavelength.
    """
    # Local imports: keep this module importable without the static-data and
    # network stack (the smoke gate imports every reader module).
    import pandas as pd
    from core import env
    from core.files import mdir
    from core.network.download import download_url

    if file_path is None:
        url = ('https://github.com/hygeos/eoread/releases/download/root/'
               'SpectralResponses_GEOSAT-2.xlsx')
        file_path = download_url(url, dirname=mdir(env.getdir('DIR_STATIC') / 'srf' / 'geosat'))

    # first sheet is de2 (GEOSAT-2)
    if sensor == 'DE-1':
        sheet = 1
        num_bands = 3
    elif sensor == 'DE-2':
        sheet = 0
        num_bands = 5
    else:
        raise ValueError(f'unknown GEOSAT sensor {sensor!r} (expected DE-1 or DE-2)')

    df = pd.read_excel(file_path, sheet_name=sheet, header=None)

    # Band names on the first row, odd columns starting at index 1.
    band_names = df.iloc[0, 1::2].values
    band_names = [name for name in band_names if not (pd.isna(name) or name == '')][:num_bands]

    # Normalize band names to match DIMAP band descriptions (case-sensitive):
    # GEOSAT-1 SRF uses "D1 NIR", "D1 RED", "D1 GREEN" -> DIMAP: "NIR", "Red", "Green".
    BAND_NAME_MAP = {'D1 NIR': 'NIR', 'D1 RED': 'Red', 'D1 GREEN': 'Green'}

    data_vars = {}
    coords = {}
    band_central_wavelengths = {}
    for i, raw_name in enumerate(band_names):
        if not panchromatic and str(raw_name).upper() == 'PAN':
            continue

        band_name = str(raw_name).strip()
        band_name = BAND_NAME_MAP.get(band_name, band_name)

        # Wavelengths for this band (even columns: 0, 2, 4, ...); SRF data in
        # the adjacent odd column.
        wav_raw = pd.to_numeric(df.iloc[1:, 2 * i], errors='coerce').values
        srf_raw = pd.to_numeric(df.iloc[1:, 2 * i + 1], errors='coerce').values

        # Drop rows where either wavelength or SRF is NaN (varying-length bands).
        valid = ~np.isnan(wav_raw) & ~np.isnan(srf_raw)
        wavelengths = wav_raw[valid]
        band_data = srf_raw[valid].tolist()

        coord_name = f'wav_{band_name}'
        coords[coord_name] = wavelengths
        data_vars[band_name] = ([coord_name], band_data)

        assert len(band_data) == len(wavelengths)

        # SRF-weighted average wavelength for this band.
        srf = np.array(band_data, dtype=float)
        srf_sum = srf.sum()
        if srf_sum > 0:
            band_central_wavelengths[band_name] = float((wavelengths * srf).sum() / srf_sum)
        else:
            band_central_wavelengths[band_name] = float(wavelengths.mean())

    # Sort band names by increasing weighted-average wavelength.
    sorted_band_names = sorted(data_vars.keys(), key=lambda b: band_central_wavelengths[b])

    ds = xr.Dataset(data_vars, coords=coords)
    for coord in ds.coords:
        ds[coord].attrs.update(units='nm')

    return ds[sorted_band_names]


def _gcp_block(block, block_info, spline):
    """Evaluate a GCP RectBivariateSpline on the pixel centers of one block.

    `block_info` is auto-injected by dask (the parameter name is the contract);
    key 0 locates the first input array, whose block layout gives the absolute
    row/column offsets of this block in the full grid.
    """
    r0 = block_info[0]['array-location'][0][0]
    c0 = block_info[0]['array-location'][1][0]
    rows = np.arange(r0, r0 + block.shape[0], dtype='float64') - 0.5
    cols = np.arange(c0, c0 + block.shape[1], dtype='float64') - 0.5
    return spline(rows, cols).astype('float32')


class Level1_GEOSAT(Reader):
    """GEOSAT-1 / GEOSAT-2 multi-spectral L1 radiance products."""

    sensor = 'GEOSAT'
    level = 'L1'
    chunks_default = 1024
    payload_var = str(names.ltoa)
    required_vars = (str(names.sza), str(names.saa),
                     str(names.vza), str(names.vaa))
    sample_products = ('GEOSAT_DE1', 'GEOSAT_DE2')
    srf_getter = 'eoread.readers.geosat.read_geosat_srf'

    VARIANTS = {
        'DE-1': dict(detect=dict(
            field=('Dataset_Sources', 'Source_Information', 'Scene_Source', 'MISSION'),
            value='DEIMOS')),
        'DE-2': dict(detect=dict(
            field=('Dataset_Sources', 'Source_Information', 'Scene_Source', 'MISSION'),
            value='Deimos 2')),
    }

    # DIMAP band description -> central wavelength (nm); case-sensitive, as in
    # the legacy reader.
    _BAND_WAVELENGTHS = {'NIR': 830, 'Red': 660, 'RED': 660, 'Green': 560,
                         'GREEN': 560, 'Blue': 490, 'BLUE': 490}

    # --------------------------------------------------------------- read

    def read(self, path: Path) -> xr.Dataset:
        product = Path(path).resolve()
        assert product.is_dir(), f'GEOSAT product not found: {product}'

        if self.verbose:
            log.debug(f'Reading GEOSAT L1 product {product.name}')

        # Flat product layout: one .dim and one .tif at the top level.
        dim_paths = sorted(product.glob('*.dim'))
        tif_paths = sorted(product.glob('*.tif'))
        assert len(dim_paths) == 1 and len(tif_paths) == 1, (
            f'expected one .dim and one .tif in {product}, '
            f'got {[p.name for p in dim_paths]} / {[p.name for p in tif_paths]}')

        dim = read_xml(str(dim_paths[0]))
        variant = self.detect_variant(dim)
        scene = dim['Dataset_Sources']['Source_Information']['Scene_Source']

        # Band table, in file band order.
        bands_info = dim['Image_Interpretation']['Spectral_Band_Info']
        if isinstance(bands_info, dict):
            bands_info = [bands_info]
        bands = [str(s['BAND_DESCRIPTION']) for s in bands_info]
        gains = np.array([_num(s['PHYSICAL_GAIN']) for s in bands_info], dtype='float64')
        bias = np.array([_num(s['PHYSICAL_BIAS']) for s in bands_info], dtype='float64')
        # Radiance unit as stated by the product (never hardcoded).
        unit = xml_value(bands_info[0].get('PHYSICAL_UNIT'))

        cwav = np.array([self._BAND_WAVELENGTHS[b] for b in bands], dtype='float64')

        # Legacy reversed the file band order; the stable ascending sort of
        # central wavelengths reproduces that exact order for both variants.
        order = list(np.argsort(cwav, kind='stable'))
        bands = [bands[i] for i in order]
        cwav = cwav[order]
        gains = gains[order]
        bias = bias[order]

        with rasterio.open(str(tif_paths[0])) as src:
            crs = str(src.crs) if src.crs else 'EPSG:4326'
            transform = src.transform
            gcps, _ = src.gcps
        gcp_pts = [(float(g.row), float(g.col), float(g.x), float(g.y)) for g in gcps]

        dn = self._open_tif(tif_paths[0], variant)
        ny, nx = int(dn.sizes[str(names.rows)]), int(dn.sizes[str(names.columns)])

        # GEOSAT-1: scattered GCPs gridded and splined (legacy algorithm).
        # GEOSAT-2: UTM affine grid transformed lazily, block by block.
        if variant == 'DE-1':
            assert len(gcp_pts) >= 4, \
                f'a bicubic GCP grid needs at least 4 points, got {len(gcp_pts)}'
            lat_spline, lon_spline = self._gcp_splines(gcp_pts)
            chunks = (self.chunks[str(names.rows)], self.chunks[str(names.columns)])
            dummy = da.zeros((ny, nx), chunks=chunks)
            lat = da.map_blocks(_gcp_block, dummy, spline=lat_spline, dtype='float32')
            lon = da.map_blocks(_gcp_block, dummy, spline=lon_spline, dtype='float32')
        else:
            ll = latlon_utm(crs, transform.c, transform.f, transform.a, transform.e,
                            nx, ny, self.chunks)
            # .data: the tuple-form assignment below needs raw arrays, and the
            # canonical y/x index coords are assigned at the end of read().
            lat = ll['latitude'].data
            lon = ll['longitude'].data

        # Sun/viewing angles are constant over the scene (flat terrain).
        cy, cx = self.chunks[str(names.rows)], self.chunks[str(names.columns)]

        def const(value: float):
            return da.full((ny, nx), value, dtype='float32', chunks=(cy, cx))

        vza_val = float(_num(scene['INCIDENCE_ANGLE']))
        assert vza_val < 20, f'GEOSAT vza {vza_val} out of range (expected < 20)'
        sza = const(90.0 - float(xml_value(scene['SUN_ELEVATION'])))
        saa = const(float(xml_value(scene['SUN_AZIMUTH'])))
        vza = const(vza_val)
        vaa = const(0.0)

        # Legacy calibration: divide by gain when the first gain is > 1 (DE-1
        # radiance scale), multiply otherwise (DE-2 reflectance scale); float64
        # math, cast to float32 at the end for legacy bit parity.
        # Reorder the payload to match the cwav-sorted band table (the file
        # bands are not in wavelength order).
        dn64 = dn.isel(bands=order).astype('float64')
        g = xr.DataArray(gains, dims=(str(names.bands),))
        b = xr.DataArray(bias, dims=(str(names.bands),))
        ltoa = (dn64 / g + b) if gains[0] > 1 else (dn64 * g + b)
        ltoa = ltoa.astype('float32')

        ds = xr.Dataset({self.payload_var: ltoa})
        ds = ds.assign_coords({str(names.bands): bands,
                               str(names.rows): np.arange(ny),
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

        if variant == 'DE-1':
            datetime_ = f'{scene["IMAGING_DATE"]}T{scene["IMAGING_TIME"]}'
            resolution = float(_num(scene['THEORETICAL_RESOLUTION']))
        else:
            # Legacy produced a date-only value for DE-2 (no IMAGING_TIME);
            # the full START_TIME is the documented fix.
            datetime_ = str(xml_value(scene['START_TIME']))
            resolution = float(_num(scene['PIXEL_RESOLUTION_X']))

        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): variant,
            str(names.product_name): str(dim['Dataset_Id']['DATASET_NAME']),
            str(names.input_directory): str(product.parent),
            str(names.datetime): datetime_,
            str(names.resolution): resolution,
            str(names.shortname): str(dim['Production']['PRODUCT_TYPE']),
            str(names.crs): crs,
            '_srf_getter': self.srf_getter,
            '_srf_getter_arg': variant,
            'metadata': metadata,
        })
        if unit is not None:
            ds[self.payload_var].attrs['unit'] = str(unit)

        return self.finalize(ds)

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _gcp_splines(gcp_pts) -> tuple:
        """Build the legacy GCP interpolation splines (lat, lon).

        The GCPs are scattered; legacy gridded them by nearest unique row/col
        (zero-filled), then fit a bicubic RectBivariateSpline per coordinate.
        `gcp_pts` is a list of (row, col, lon, lat) tuples.
        """
        rows_u = np.sort(np.unique([p[0] for p in gcp_pts]))
        cols_u = np.sort(np.unique([p[1] for p in gcp_pts]))
        grids = {'lat': np.zeros((len(rows_u), len(cols_u))),
                 'lon': np.zeros((len(rows_u), len(cols_u)))}
        for row, col, lon, lat in gcp_pts:
            i = int(np.argmin(np.abs(rows_u - row)))
            j = int(np.argmin(np.abs(cols_u - col)))
            grids['lat'][i, j] = lat
            grids['lon'][i, j] = lon
        splines = {k: RectBivariateSpline(rows_u, cols_u, g, kx=3, ky=3)
                   for k, g in grids.items()}
        return splines['lat'], splines['lon']

    def _open_tif(self, path: Path, variant: str) -> xr.DataArray:
        """Open the GeoTIFF lazily as float32 DN; nodata masked for DE-2."""
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
        dn = arr.astype('float32')
        if variant == 'DE-2':
            # Nodata 0 -> NaN (documented deviation: legacy kept the zeros).
            dn = dn.where(dn != 0)
        return dn
