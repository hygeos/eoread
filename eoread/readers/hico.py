#!/usr/bin/env python3
"""
ISS HICO (Hyperspectral Imager for the Coastal Ocean) reader, level L1B.

    >>> from eoread.readers.hico import Level1_HICO       # doctest: +SKIP
    >>> ds = Level1_HICO('H2014069115230.L1B_ISS.nc')     # doctest: +SKIP

HICO is a hyperspectral imaging spectrometer that flew on the International
Space Station (2011-2018). The OB.DAAC Level-1B "ISS" products are netCDF-4
files holding top-of-atmosphere radiance in 128 bands (353-1080 nm) on a
~100 m tiled grid, with per-pixel geolocation and solar/viewing angles.

File layout:
    root attrs                      processing_level, resolution, ...
    navigation/                     latitudes, longitudes, solar/sensor
                                    zenith and azimuth (scan_lines, samples)
    products/Lt                     radiance (scan_lines, samples, bands),
                                    uint16 with CF scale_factor decoding
    scan_line_attributes/           scan_quality_flags (entirely fill in the
                                    OB.DAAC ISS L1B generation)
    metadata/FGDC/...               FGDC metadata, incl. Time_Period_of_Content

Documented deviations from the v1 implementation (see REFACTORING.md):
- Canonical orientation: the stored row order is the scan order, so which
  edge is north depends on the orbit; v2 flips per file to south-up /
  west-to-east (v1 kept the native layout).
- Per-band chunking where v1 read one all-bands block.
- `resolution` from the product's own global attribute ('100m' -> 100)
  instead of v1's hardcoded 30 (the sample grid measures ~95-105 m).
- `shortname` / `crs` attributes added (schema requirements; v1 did not set them).
- Azimuths keep the product's absolute [0, 360) convention (registry declares
  [0, 180] - documented registry inconsistency, see MERIS/OLCI).
- Constructor option `engine` (default 'netcdf4') selects the xarray I/O
  backend; v1 used the xarray default (effectively netcdf4).
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from core.geo.naming import names
from core import log

from eoread.flags import GenericFlags
from eoread.reader import Reader
from eoread.tools import filter_metadata


# Nominal central wavelengths (nm) of the HICO bands, in product order.
# These identifiers are shared with NASA's iss_hico RSR table
# (https://oceandata.sci.gsfc.nasa.gov/images/rsr/iss_hico_RSR.nc) and with
# polymer's `bands_hico` list.
HICO_BANDS = [
    353, 358, 364, 370, 375, 381, 387, 393, 398, 404, 410,
    416, 421, 427, 433, 438, 444, 450, 456, 461, 467, 473,
    479, 484, 490, 496, 501, 507, 513, 519, 524, 530, 536,
    542, 547, 553, 559, 564, 570, 576, 582, 587, 593, 599,
    605, 610, 616, 622, 627, 633, 639, 645, 650, 656, 662,
    668, 673, 679, 685, 690, 696, 702, 708, 713, 719, 725,
    731, 736, 742, 748, 753, 759, 765, 771, 776, 782, 788,
    794, 799, 805, 811, 816, 822, 828, 834, 839, 845, 851,
    857, 862, 868, 874, 880, 885, 891, 897, 902, 908, 914,
    920, 925, 931, 937, 943, 948, 954, 960, 965, 971, 977,
    983, 988, 994, 1000, 1006, 1011, 1017, 1023, 1028, 1034, 1040,
    1046, 1051, 1057, 1063, 1069, 1074, 1080,
]


class Level1_HICO(Reader):
    """ISS/HICO Level-1B product (OB.DAAC "L1B_ISS" tiles).

    Reads the TOA radiance cube (``Lt``, CF-decoded at open) and the
    per-pixel navigation group, renamed onto the canonical names. Band
    labels are the nominal central wavelengths (as strings); the actual
    central wavelengths are stored in ``cwav``.

    No usable per-pixel quality flags exist in this product generation:
    ``scan_quality_flags`` is entirely fill (255), and QC is applied by the
    Level-2 processor — so every standard flag decodes to an all-False mask
    (v1 / polymer v4 parity).

    Constructor option `engine` (default 'netcdf4') selects the xarray I/O
    backend for the open — see `__init__` for the rationale.
    """

    sensor = 'HICO'
    level = 'L1B'
    chunks_default = 500
    payload_var = str(names.ltoa)
    required_vars = (str(names.sza), str(names.saa), str(names.vza),
                     str(names.vaa), 'altitude')
    sample_products = ('HICO_L1B',)
    flags = True
    VARIANTS = {
        'l1b': dict(detect=dict(field='processing_level', value='L1B')),
    }

    def __init__(self, path, chunks=None, metadata_template=None, verbose=False,
                 engine='netcdf4'):
        # `engine` selects the xarray I/O backend for the open. Default
        # 'netcdf4' is the engine the v1 reader effectively used (xarray
        # default). Pass engine='h5netcdf' when the process must stay on a
        # single libhdf5 copy — i.e. it also uses h5py-based readers or
        # writers: netCDF4's bundled libhdf5 poisons their dimension-scale
        # operations (REFACTORING.md, dual-libhdf5 decision).
        super().__init__(path, chunks=chunks, metadata_template=metadata_template,
                         verbose=verbose)
        self.engine = engine

    def read(self, path: Path) -> xr.Dataset:
        path = Path(path)
        assert path.exists(), f'{path} does not exist'

        if self.verbose:
            log.debug(f'Reading HICO {self.level} product {path.name}')

        tree = xr.open_datatree(path, engine=self.engine)

        # Loud on a non-L1B product (root global attribute).
        self.detect_variant(dict(tree.attrs))

        products = tree['products'].to_dataset()
        nav = tree['navigation'].to_dataset()
        tperiod = (tree['metadata']['FGDC']['Identification_Information']
                   ['Time_Period_of_Content'])

        # Loud on an unexpected payload.
        lt = products['Lt']
        assert str(lt.attrs.get('units')) == 'W/m^2/micrometer/sr', \
            f'Unexpected units for Lt: {lt.attrs.get("units")}'
        wavelengths = np.asarray(lt.attrs['wavelengths'], dtype='float32')
        assert len(wavelengths) == len(HICO_BANDS), \
            f'Expected {len(HICO_BANDS)} bands, got {len(wavelengths)}'

        rows, cols, bands = str(names.rows), str(names.columns), str(names.bands)
        ds = nav.rename({
            'latitudes': str(names.lat), 'longitudes': str(names.lon),
            'solar_zenith': str(names.sza), 'sensor_zenith': str(names.vza),
            'solar_azimuth': str(names.saa), 'sensor_azimuth': str(names.vaa)})
        # CF-decoded at open time (scale_factor applied; uint16 -> float32)
        ds[str(names.ltoa)] = lt
        ds[str(names.ltoa)].attrs['units'] = 'W/m^2/micrometer/sr'
        ds = ds.rename(scan_lines=rows, samples=cols)

        # One all-bands block (legacy chunking): the file stores Lt as a single
        # contiguous uncompressed volume, so per-band blocks turn every read
        # into a strided pass over the whole swath (1024 reads vs 8 for a full
        # load on the sample). Chunking before the orientation flip: a
        # negative-step isel over the raw h5netcdf backend array cannot be
        # decomposed by xarray, while dask-backed variables handle it
        # transparently.
        ds = ds.chunk({bands: -1, rows: self.chunks[rows], cols: self.chunks[cols]})

        # Canonical orientation (the sample is a north-up descending granule;
        # the flip is decided per file, as for ECOSTRESS).
        ds = self._orient(ds)

        # Spectral coordinates: nominal band labels, actual central
        # wavelengths, single group.
        ds = ds.assign_coords({bands: [str(b) for b in HICO_BANDS]})
        ds[str(names.cwav)] = (bands, wavelengths,
                               {'units': 'nm', 'desc': 'Central wavelength'})
        ds = ds.assign_coords({str(names.bgroup): (bands, ['bands'] * len(HICO_BANDS))})

        # Altitude: HICO products carry no terrain information (polymer v4
        # also processed HICO with a zero altitude)
        ds['altitude'] = xr.zeros_like(ds[str(names.lat)], dtype='float32')
        ds['altitude'].attrs.update({'units': 'm', 'desc': 'Terrain altitude'})

        # eoread convention: 3D rasters in bands-first order
        ds = ds.transpose(..., rows, cols)

        # Attributes (legacy parity + schema requirements); the metadata copy
        # is filtered when a template is given.
        datetime_str = (f"{tperiod.attrs['Beginning_Date']}"
                        f"{tperiod.attrs['Beginning_Time']}")
        res = str(tree.attrs.get('resolution', ''))
        match = re.search(r'\d+', res)
        assert match, f'cannot parse a resolution from product attribute {res!r}'

        # Product class from the OB.DAAC file name (H<date><time>.<CLASS>.nc)
        parts = path.name.split('.')
        assert len(parts) == 3 and parts[1].startswith('L'), \
            f'unexpected HICO file name {path.name!r}'

        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): 'ISS',
            str(names.product_name): path.name,
            str(names.input_directory): str(path.parent),
            str(names.datetime): datetime.strptime(datetime_str, '%Y%m%d%H%M%S').isoformat(),
            str(names.resolution): int(match.group()),
            str(names.shortname): parts[1],
            str(names.crs): 'EPSG:4326',
            'metadata': dict(tree.attrs) if self.metadata_template is None
                        else filter_metadata(dict(tree.attrs), self.metadata_template),
        })
        # Spectral response functions from NASA OB.DAAC (iss_hico_RSR.nc)
        ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_NASA'
        ds.attrs['_srf_getter_arg'] = 'iss_hico'

        return self.finalize(ds)

    # --------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return []

    @classmethod
    def flag_dims_like(cls) -> str:
        # Per-pixel flags are 2D (a 3D reference such as Ltoa would give
        # per-band flags, which is not what downstream processors expect)
        return str(names.lat)

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """All standard flags are False: the product carries no usable QC.

        The file does contain a ``scan_quality_flags`` variable, but it is
        entirely fill (255) in the OB.DAAC ISS L1B generation — quality
        control is applied by the Level-2 processor. Matches v1 / polymer v4.
        """
        return xr.zeros_like(ds[cls.flag_dims_like()], dtype='bool')

    # ---------------------------------------------------------- orientation

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
