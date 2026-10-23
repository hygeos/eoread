#!/usr/bin/env python3
"""
PlanetScope PSScene Basic Analytic 8b reader, level L1 (reflectance).

    >>> from eoread.readers.planetscope import Level1_PLANETSCOPE   # doctest: +SKIP
    >>> ds = Level1_PLANETSCOPE('PSScene_Basic_Analytic_8b_udm2')   # doctest: +SKIP

Differences from the legacy v1 readers (documented deviations, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the scene (canonical
  orientation). Legacy outputs were north-up.
- Nodata pixels (DN == 0) become NaN; the legacy rioxarray open kept them as
  zero data (masked=False default).
- `cwav` is float64 (legacy float32); the payload, angles and lat/lon are
  float32 (legacy computed the constant angles in float64).
"""

import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from core import log
from core.geo.naming import names
from core.table import read_xml

from eoread.reader import Reader
from eoread.tools import (_blockwise_interp, filter_metadata,
                          install_exit_cleanup, xml_value)


def read_srf_superdove(srf_file: Path | None = None) -> xr.Dataset:
    """
    Read the PlanetScope (SuperDove) spectral response functions.

    One variable per band on a shared `wavelength` coordinate in nm. The CSV
    is downloaded from the eoread release assets and cached under DIR_STATIC
    on first use.

    Parameters
    ----------
    srf_file : Path, optional
        Path to a semicolon-separated CSV with one column per band (labelled
        by central wavelength) plus `Wavelength (nm)`. Downloaded when None.
    """
    # Local imports: keep this module importable without the static-data and
    # network stack (the smoke gate imports every reader module).
    import pandas as pd
    from core import env
    from core.files import mdir
    from core.network.download import download_url

    if srf_file is None:
        url = 'https://github.com/hygeos/eoread/releases/download/root/Superdove_SRF.csv'
        srf_file = download_url(url, mdir(env.getdir('DIR_STATIC') / 'srf' / 'planetscope'))

    df = pd.read_csv(srf_file, sep=';')

    # Legacy band labels (central wavelength in nm); the CSV columns carry
    # exactly these names.
    band_names = ['441', '490', '531', '565', '610', '665', '705', '865']

    ds = xr.Dataset()
    for band in band_names:
        ds[band] = xr.DataArray(df[band].values, dims=['wavelength'])

    ds = ds.assign_coords(wavelength=df['Wavelength (nm)'].values)

    # Remove negative/invalid values
    ds = ds.where(ds > 0, 0)

    ds.wavelength.attrs.update(units='nm')

    return ds


def _rpc_block(block, block_info, rpc_dict, which):
    """Apply the RPC model to one block of pixel indices (lazy geolocation).

    `block_info` is auto-injected by dask (the parameter name is the contract);
    key 0 locates the first input array, whose block layout gives the absolute
    row/column offsets of this block in the full grid. The rasterio RPC and
    transformer objects are not picklable, so the plain coefficient dict is
    passed as a kwarg and the transformer is rebuilt inside every block.
    """
    from rasterio.rpc import RPC
    from rasterio.transform import RPCTransformer

    r0 = block_info[0]['array-location'][0][0]
    c0 = block_info[0]['array-location'][1][0]
    lines = np.arange(r0, r0 + block.shape[0], dtype='float64')
    samps = np.arange(c0, c0 + block.shape[1], dtype='float64')
    line_g, samp_g = np.meshgrid(lines, samps, indexing='ij')

    rpc = RPC(
        line_off=rpc_dict['LINE_OFF'],
        line_scale=rpc_dict['LINE_SCALE'],
        line_num_coeff=[rpc_dict[f'LINE_NUM_COEFF_{i}'] for i in range(1, 21)],
        line_den_coeff=[rpc_dict[f'LINE_DEN_COEFF_{i}'] for i in range(1, 21)],
        samp_off=rpc_dict['SAMP_OFF'],
        samp_scale=rpc_dict['SAMP_SCALE'],
        samp_num_coeff=[rpc_dict[f'SAMP_NUM_COEFF_{i}'] for i in range(1, 21)],
        samp_den_coeff=[rpc_dict[f'SAMP_DEN_COEFF_{i}'] for i in range(1, 21)],
        lat_off=rpc_dict['LAT_OFF'],
        lat_scale=rpc_dict['LAT_SCALE'],
        long_off=rpc_dict['LONG_OFF'],
        long_scale=rpc_dict['LONG_SCALE'],
        height_off=rpc_dict['HEIGHT_OFF'],
        height_scale=rpc_dict['HEIGHT_SCALE'],
    )

    # transformer.xy returns (lon, lat) as flattened arrays.
    lon, lat = RPCTransformer(rpc).xy(line_g, samp_g, zs=0.0)
    out = np.asarray(lat if which == 'lat' else lon).reshape(block.shape)
    return out.astype('float32')


class Level1_PLANETSCOPE(Reader):
    """PlanetScope PSScene Basic Analytic 8b L1 reflectance products."""

    sensor = 'PLANETSCOPE'
    level = 'L1'
    chunks_default = 1024
    payload_var = str(names.rtoa)
    required_vars = (str(names.sza), str(names.saa),
                     str(names.vza), str(names.vaa))
    sample_products = ('PLANETSCOPE_MS',)
    srf_getter = 'eoread.readers.planetscope.read_srf_superdove'

    # Central wavelengths in nm, file band order (already ascending).
    _CWAWS = np.array([443, 490, 531, 565, 610, 665, 705, 865], dtype='float64')

    # --------------------------------------------------------------- read

    def read(self, path: Path) -> xr.Dataset:
        product = Path(path).resolve()
        assert product.is_dir(), f'PlanetScope product not found: {product}'

        if self.verbose:
            log.debug(f'Reading PlanetScope L1 product {product.name}')

        # The product root contains one 8b tif and one OGC XML somewhere in
        # its tree (the pan tif must not match).
        tif_paths = sorted(product.glob('**/*8b.tif'))
        xml_paths = sorted(product.glob('**/*.xml'))
        assert len(tif_paths) == 1, (
            f'expected exactly one *8b.tif in {product}, '
            f'got {[p.name for p in tif_paths]}')
        assert len(xml_paths) == 1, (
            f'expected exactly one XML in {product}, '
            f'got {[p.name for p in xml_paths]}')

        ogc = read_xml(str(xml_paths[0]))

        dn = self._open_tif(tif_paths[0])
        bands = [str(b) for b in dn[str(names.bands)].values]
        ny, nx = int(dn.sizes[str(names.rows)]), int(dn.sizes[str(names.columns)])
        assert len(bands) == 8, f'expected an 8-band MS product, got {len(bands)} bands'

        # Reflectance coefficients in file band order (bandNumber - 1).
        info = ogc['resultOf']['EarthObservationResult']['bandSpecificMetadata']
        if isinstance(info, dict):
            info = [info]
        coef = np.zeros(len(bands), dtype='float32')
        for entry in info:
            idx = int(entry['bandNumber']) - 1
            assert 0 <= idx < len(bands), f'bandNumber {idx + 1} out of range'
            coef[idx] = float(xml_value(entry['reflectanceCoefficient']))

        # All-float32 chain: DN * coefficient (no /cos(sza), as legacy).
        rtoa = dn * xr.DataArray(coef, dims=(str(names.bands),))
        rtoa = rtoa.astype('float32')

        lat, lon = self._latlon(ogc, product, ny, nx)

        # Sun/viewing angles are constant over the scene (flat terrain).
        cy, cx = self.chunks[str(names.rows)], self.chunks[str(names.columns)]
        acq = (ogc['using']['EarthObservationEquipment']
               ['acquisitionParameters']['Acquisition'])

        def const(value: float):
            return da.full((ny, nx), value, dtype='float32', chunks=(cy, cx))

        sza = const(90.0 - float(xml_value(acq['illuminationElevationAngle'])))
        saa = const(float(xml_value(acq['illuminationAzimuthAngle'])))
        vza = const(float(xml_value(acq['spaceCraftViewAngle'])))
        vaa = const(float(xml_value(acq['azimuthAngle'])))

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
        ds[str(names.cwav)] = (str(names.bands), self._CWAWS.copy())
        ds[str(names.cwav)].attrs['unit'] = 'nm'
        ds[str(names.bgroup)] = (str(names.bands), ['bands_vnir'] * len(bands))
        ds = ds.set_coords(str(names.bgroup))

        # Curated attributes (legacy parity + schema requirements)
        if self.metadata_template is None:
            metadata = ogc
        else:
            metadata = filter_metadata(ogc, self.metadata_template)

        platform = (ogc['using']['EarthObservationEquipment']
                    ['platform']['Platform'])
        meta = ogc['metaDataProperty']['EarthObservationMetaData']

        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): str(xml_value(platform['shortName'])),
            str(names.product_name): product.name,
            str(names.input_directory): str(product.parent),
            str(names.datetime): str(xml_value(
                meta['downlinkedTo']['DownlinkInformation']['acquisitionDate'])),
            str(names.resolution): 3,
            str(names.shortname): str(xml_value(meta['productType'])),
            str(names.crs): 'EPSG:4326',
            '_srf_getter': self.srf_getter,
            '_srf_getter_arg': None,
            'metadata': metadata,
        })
        return self.finalize(ds)

    # ------------------------------------------------------------- helpers

    def _open_tif(self, path: Path) -> xr.DataArray:
        """Open the 8b tif lazily as float32 DN with nodata masked.

        rioxarray (not xarray's rasterio backend) because the band
        descriptions come back as `long_name` attributes, which are the
        canonical band labels for this product (legacy parity).
        """
        import rioxarray

        chunks = {'y': self.chunks[str(names.rows)],
                  'x': self.chunks[str(names.columns)]}
        with warnings.catch_warnings():
            # xarray's chunk normalizer warns when our chunk boundaries fall
            # inside the file's stored tiles (one-shot windowed read; the
            # user-facing chunk spec takes precedence, so the hint is noise).
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message='The specified chunks separate the stored chunks')
            arr = rioxarray.open_rasterio(str(path), chunks=chunks)
        install_exit_cleanup()
        arr = arr.rename({'band': str(names.bands)})
        labels = list(arr.attrs.get('long_name', ()))
        assert len(labels) == arr.sizes[str(names.bands)], \
            f'expected {arr.sizes[str(names.bands)]} band descriptions, got {len(labels)}'
        arr = arr.assign_coords({str(names.bands): [str(b) for b in labels]})
        # The file y/x coords are pixel centers (0.5, 1.5, ...); canonical
        # pixel indices are assigned at the end of read().
        for coord in ('x', 'y', 'spatial_ref'):
            if coord in arr.coords:
                arr = arr.drop_vars(coord)
        dn = arr.astype('float32')
        # Nodata 0 -> NaN (documented deviation: legacy kept the zeros).
        return dn.where(dn != 0)

    def _latlon(self, ogc: dict, product: Path, ny: int, nx: int) -> tuple:
        """Lat/lon grid from the RPC model, or the four footprint corners.

        File space (row 0 = north), computed lazily. The RPC path evaluates
        one task per output block; the corner fallback is bit-for-bit
        identical to a full-grid core.interpolate.interp evaluation.
        """
        chunks = (self.chunks[str(names.rows)], self.chunks[str(names.columns)])

        rpc_paths = sorted(product.glob('**/*_RPC.TXT'))
        if rpc_paths:
            # 0-based pixel indices, as in the legacy reader (no +0.5).
            dummy = da.zeros((ny, nx), chunks=chunks)
            rpc_dict = self._parse_rpc(rpc_paths[0])
            lat = da.map_blocks(_rpc_block, dummy, rpc_dict=rpc_dict,
                                which='lat', dtype='float32')
            lon = da.map_blocks(_rpc_block, dummy, rpc_dict=rpc_dict,
                                which='lon', dtype='float32')
            return lat, lon

        geo = ogc['target']['Footprint']['geographicLocation']
        lat_g = np.array([[float(geo['topLeft']['latitude']),
                           float(geo['topRight']['latitude'])],
                          [float(geo['bottomLeft']['latitude']),
                           float(geo['bottomRight']['latitude'])]], dtype='float32')
        lon_g = np.array([[float(geo['topLeft']['longitude']),
                           float(geo['topRight']['longitude'])],
                          [float(geo['bottomLeft']['longitude']),
                           float(geo['bottomRight']['longitude'])]], dtype='float32')
        src = ([0.0, ny - 1.0], [0.0, nx - 1.0])
        targets = (np.arange(ny, dtype='float64'), np.arange(nx, dtype='float64'))
        lat = _blockwise_interp(lat_g, src, targets, chunks).astype('float32')
        lon = _blockwise_interp(lon_g, src, targets, chunks).astype('float32')
        return lat, lon

    @staticmethod
    def _parse_rpc(path: Path) -> dict:
        """Parse a PlanetScope RPC text file (``KEY: value`` lines)."""
        rpc_dict = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                rpc_dict[key.strip()] = float(value.strip())
        return rpc_dict
