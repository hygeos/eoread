#!/usr/bin/env python3
"""
MERIS (Medium Resolution Imaging Spectrometer) readers, level L1.

`Level1_MERIS` reads ESA N1 products (epr); `Level1_MERIS_SAFE` reads the
NASA-repackaged NetCDF SAFE (`.SEN3`) products (see its docstring):

    >>> from eoread.readers.meris import Level1_MERIS       # doctest: +SKIP
    >>> ds = Level1_MERIS('MER_RR__1PRACR...N1')            # doctest: +SKIP
    >>> from eoread.readers.meris import Level1_MERIS_SAFE  # doctest: +SKIP
    >>> ds = Level1_MERIS_SAFE('ENV_ME_1_RRG____....SEN3')  # doctest: +SKIP

Differences from the legacy v1 reader (documented deviations, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the scene (canonical
  orientation). Legacy outputs were north-up.
- ``F0`` / ``wav`` are always emitted (legacy gated them behind
  ``read_auxdata=False``): per-pixel solar spectral flux and central
  wavelength, gathered lazily from the per-detector characterization tables
  and NaN where ``detector_index < 0``.
- ``resolution`` is read from the SPH ``COLUMN_SPACING`` field (across-track
  GSD; 1040 m on the RR sample) instead of legacy's hardcoded 300 m.
- The ``shortname`` and ``crs`` attributes are now set (legacy did not).
- Azimuth angles (``saa`` / ``vaa``) keep the product's absolute [0, 360)
  convention; the naming registry declares [0, 180] for every azimuth — a
  documented registry inconsistency (see the OLCI/MSI entries in
  REFACTORING.md).
- Single-element metadata bytes fields are decoded to ``str``.

The epr C API is not thread-safe; every band read goes through one shared
`Lock` (parallel dask workers share a single product). The product is never
explicitly closed — the interpreter exits cleanly with it held open (pinned
by the exit-probe test), so no atexit cleanup is needed.
"""

import re
import warnings
from datetime import datetime
from pathlib import Path
from threading import Lock

import dask.array as da
import epr
import numpy as np
import xarray as xr

from core import log
from core.geo.naming import names
from core.table import read_xml
from core.tools import getflag

from eoread.flags import GenericFlags
from eoread.reader import ProductError, Reader, UnsupportedProductType
from eoread.transforms.angles import tiepoint_angles
from eoread.tools import filter_metadata, spatial_resample

#: User guide for the MERIS L1 product (legacy attribute kept).
USER_GUIDE = 'https://archive.org/details/manualzilla-id-5933919/page/n13/mode/2up'

#: Per-detector characterization tables, shipped with the v2 package.
_AUXDATA_DIR = Path(__file__).resolve().parents[1] / 'auxdata' / 'meris'


def _decode(value):
    """Decode EPR bytes fields to str; other values (ints, arrays) pass through."""
    return value.decode('utf-8') if isinstance(value, bytes) else value


def _read_date(dat: bytes | str) -> datetime:
    """Parse a MERIS date string ('01-JUL-2008 01:40:28.713088')."""
    dat = _decode(dat)
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for i, m in enumerate(months):
        dat = dat.replace(f'-{m}-', f'-{i + 1:02d}-')
    return datetime.strptime(dat, '%d-%m-%Y %H:%M:%S.%f')


class _EPRBand:
    """Array-like over one EPR band, guarded by a shared lock.

    Ported from the legacy reader; `__getitem__` maps dask block selections
    to strided `read_as_array` windows.
    """

    def __init__(self, band, lock: Lock):
        width = band.product.get_scene_width()
        height = band.product.get_scene_height()
        self.band = band
        self.lock = lock
        self.shape = (height, width)
        self.dtype = {'float': np.float32, 'short': np.int16, 'uchar': np.uint8}[
            epr.data_type_id_to_str(band.data_type)]
        self.ndim = len(self.shape)

    def __getitem__(self, keys):
        assert len(keys) == self.ndim
        start = []
        steps = []
        sizes = []
        sel = []
        for k in keys:
            if isinstance(k, slice):
                st = k.start or 0
                start.append(st)
                steps.append(k.step or 1)
                sizes.append(k.stop - st)
                sel.append(slice(None))
            else:  # indexing with an int
                start.append(0)
                steps.append(1)
                sizes.append(1)
                sel.append(0)

        with self.lock:
            if (sizes[0] > steps[0]) and (sizes[1] > steps[1]):
                r = self.band.read_as_array(
                    yoffset=start[0], xoffset=start[1],
                    height=sizes[0], width=sizes[1],
                    ystep=steps[0], xstep=steps[1])
            else:
                r = self.band.read_as_array(
                    yoffset=start[0], xoffset=start[1],
                    height=sizes[0], width=sizes[1])
                r = r[::steps[0], ::steps[1]]
        assert r.dtype == self.dtype
        return r[sel[0], sel[1]]


class Level1_MERIS(Reader):
    """MERIS level L1 (N1 products): TOA radiance of 15 bands (payload `Ltoa`)."""

    sensor = 'MERIS'
    level = 'L1'
    chunks_default = 500
    payload_var = str(names.ltoa)
    sample_products = ('MERIS_L1',)
    required_vars = (str(names.sza), str(names.saa), str(names.vza), str(names.vaa),
                     str(names.F0), str(names.wav), 'l1_flags', 'detector_index')
    flags = True
    VARIANTS = {
        'rr': dict(detect=dict(field='PRODUCT', value='MER_RR', prefix=True)),
        'fr': dict(detect=dict(field='PRODUCT', value='MER_FR', prefix=True)),
    }

    def read(self, path: Path) -> xr.Dataset:
        path = Path(path)
        assert path.exists(), f'{path} does not exist'

        if self.verbose:
            log.debug(f'Reading MERIS {self.level} product {path.name}')

        # epr's C API is not thread-safe: one lock guards every band read
        lock = Lock()
        prod = epr.Product(str(path))
        ny, nx = prod.get_scene_height(), prod.get_scene_width()

        # Metadata (MPH + SPH; single-element bytes fields decoded to str)
        metadata = {}
        for ph in (prod.get_mph(), prod.get_sph()):
            metadata.update({
                _decode(f.get_name()):
                    _decode(f.get_elem(0)) if f.get_num_elems() == 1 else f.get_elems()
                for f in ph.fields()})

        variant = self.detect_variant(metadata)  # loud on an unknown PRODUCT prefix

        nb_bands = int(metadata['NUM_BANDS'])
        band_names = [_decode(n) for n in prod.get_band_names()]
        radiance_names = [n for n in band_names if n.startswith('radiance_')]
        assert len(radiance_names) == nb_bands, \
            f'expected {nb_bands} radiance bands, got {len(radiance_names)}'

        rows, cols, bands = str(names.rows), str(names.columns), str(names.bands)
        ds = xr.Dataset()

        # 1. Native 2D fields, lazily through the locked EPR API. Explicit
        #    chunks + meta on every band: without them dask probes each band
        #    with an empty slice at graph construction to infer the dtype, and
        #    _EPRBand maps a zero-size window to a full-scene epr read — 32
        #    full-band reads per build. Legacy passed both.
        cy = self.chunks[rows]
        cx = self.chunks[cols]
        for name in band_names:
            if name.startswith('radiance_'):
                continue  # stacked below onto the bands dim
            band = prod.get_band(name)
            eb = _EPRBand(band, lock)
            ds[name] = xr.DataArray(
                da.from_array(eb, chunks=(cy, cx), meta=np.empty((0, 0), dtype=eb.dtype)),
                dims=(rows, cols))
            if band.unit is not None:
                ds[name].attrs['unit'] = band.unit
            ds[name].attrs['description'] = str(band.description)

        # 2. Ltoa: the radiance bands stacked in product order (per-band chunks)
        arrays = []
        for n in radiance_names:
            eb = _EPRBand(prod.get_band(n), lock)
            arrays.append(da.from_array(eb, chunks=(cy, cx),
                                        meta=np.empty((0, 0), dtype=eb.dtype)))
        ltoa = xr.DataArray(da.stack(arrays), dims=(bands, rows, cols))
        ds[str(names.ltoa)] = ltoa.chunk({bands: 1, rows: cy, cols: cx})
        first = prod.get_band(radiance_names[0])
        if first.unit is not None:
            ds[str(names.ltoa)].attrs['unit'] = first.unit
        ds[str(names.ltoa)].attrs['description'] = str(first.description)

        # 3. Canonical geometry / angle names
        ds = ds.rename({'view_zenith': str(names.vza), 'view_azimuth': str(names.vaa),
                        'sun_zenith': str(names.sza), 'sun_azimuth': str(names.saa)})

        # 4. Per-pixel solar spectral flux and central wavelength, gathered
        #    lazily from the per-detector characterization tables (always
        #    emitted in v2); NaN where detector_index < 0.
        det = ds['detector_index']
        valid = det >= 0
        idx = det.where(valid, 0)
        det_da = idx.data
        for var, table_name, dtype in ((str(names.F0), 'sun_spectral_flux', 'float32'),
                                       (str(names.wav), 'central_wavelen', 'float64')):
            # The TSV carries a leading detector column; the data is per band.
            table = np.loadtxt(_AUXDATA_DIR / f'{table_name}_{variant}.txt',
                               delimiter='\t', skiprows=1, dtype=dtype)[:, 1:]
            assert table.shape[1] == nb_bands, \
                f'{table_name}_{variant} table has {table.shape[1]} bands, expected {nb_bands}'
            # `t=table` binds the per-iteration table: a bare closure would
            # late-bind and gather both variables from the last table loaded.
            # Gather as one band block — in this dask version, chunking the
            # map_blocks new axis per band inflates it to nb_bands**2 — then
            # rechunk: per-band chunks keep `Dataset.chunks` consistent with
            # Ltoa (per band).
            gathered = da.map_blocks(lambda b, t=table: t[b], det_da, dtype=dtype, new_axis=2,
                                     chunks=(det_da.chunks[0], det_da.chunks[1], (nb_bands,)))
            ds[var] = xr.DataArray(gathered.transpose(2, 0, 1),
                                   dims=(bands, rows, cols)).where(valid).chunk({bands: 1})
            ds[var].attrs['unit'] = 'mW/(m^2*sr*nm)' if var == str(names.F0) else 'nm'

        # 5. Canonical orientation: row 0 = southern edge (single vertical flip)
        ds = ds.isel({rows: slice(None, None, -1)})
        ds = ds.assign_coords({rows: np.arange(ny), cols: np.arange(nx)})

        # 6. Spectral coordinates
        cwav = np.asarray(metadata['BAND_WAVELEN'], dtype='uint32') / 1e3  # nm
        assert len(cwav) == nb_bands, \
            f'BAND_WAVELEN has {len(cwav)} entries, expected {nb_bands}'
        ds[str(names.cwav)] = (bands, cwav)
        # Scientific band names: the official MERIS band numbers B1..B15.
        ds = ds.assign_coords({bands: [f'B{i + 1}' for i in range(nb_bands)]})
        ds[str(names.bgroup)] = (bands, ['bands_vnir'] * nb_bands)
        ds = ds.set_coords(str(names.bgroup))

        # 7. Curated attributes (legacy parity + schema requirements); the
        #    metadata copy is filtered when a template is given.
        product_name = metadata['PRODUCT']
        resolution = float(metadata['COLUMN_SPACING'])
        if resolution.is_integer():
            resolution = int(resolution)

        dstart = _read_date(metadata['SENSING_START'])
        dstop = _read_date(metadata['SENSING_STOP'])
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): 'ENVISAT',
            str(names.product_name): product_name,
            str(names.input_directory): str(path.parent),
            str(names.datetime): (dstart + (dstop - dstart) // 2).isoformat(),
            str(names.resolution): resolution,
            str(names.shortname): re.split(r'\d{8}', product_name)[0],
            str(names.crs): 'EPSG:4326',
            'user_guide': USER_GUIDE,
            'metadata': metadata if self.metadata_template is None
                        else filter_metadata(metadata, self.metadata_template),
        })
        ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_meris'
        ds.attrs['_srf_getter_arg'] = ''

        return self.finalize(ds)

    # ----------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return ['l1_flags']

    @classmethod
    def flag_dims_like(cls) -> str:
        return 'l1_flags'

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """Nonzero ``l1_flags`` marks an invalid pixel."""
        if flag_name == GenericFlags.L1_INVALID:
            # Canonical bool contract; raiseflag's `condition != 0` keeps the
            # output identical to the legacy raw-value return.
            return (ds['l1_flags'] != 0)
        raise ValueError(f'Unsupported MERIS flag: {flag_name!r}')


# ---------------------------------------------------------------------------
# Level1_MERIS_SAFE (.SEN3 container)
#
# NASA repackages MERIS L1 with the same Copernicus-style `.SEN3` conventions
# as OLCI (see olci.py): an xfdumanifest.xml plus per-band and ancillary
# NetCDF files on native 'rows'/'columns' dims. The mapping constants below
# mirror that module; they are duplicated rather than imported to keep each
# reader module self-contained.
# ---------------------------------------------------------------------------

#: Native spatial dim names of the SAFE NetCDF files, before rename to the
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

#: 2D meteorological tie variables of tie_meteo.nc -> canonical output names
#: (`humidity` is a pressure-level profile and is handled separately).
_METEO_MAP = (
    ('sea_level_pressure', 'sea_level_pressure'),
    ('total_columnar_water_vapour', 'total_columnar_water_vapour'),
    ('total_ozone', 'total_column_ozone'),
)


class Level1_MERIS_SAFE(Reader):
    """MERIS level L1 (NetCDF SAFE products, `.SEN3` containers).

    NASA repackages MERIS L1 as Copernicus-style `.SEN3` directories: an
    `xfdumanifest.xml` plus per-band radiance files (`M01..M15_radiance.nc`)
    and ancillary NetCDF files on native 'rows'/'columns' dims (the same
    container conventions the OLCI reader reads — see olci.py). The N1-format
    reader above cannot open these products.

    - Variants RR / FR are detected from the manifest `productType`
      (`ME_1_RRG___` / `ME_1_FRG___`); the product level is cross-checked
      against the content-unit `textInfo`.
    - `resolution` comes from the manifest `acSpatialSampling` (1040 m RR,
      260 m FR) — data-driven, no hardcoded GSD.
    - Azimuth angles (`saa` / `vaa`) are signed degrees in [-180, 180]
      (unlike the N1 products' absolute [0, 360)); the naming registry
      declares [0, 180] for every azimuth — a documented registry
      inconsistency (REFACTORING.md).
    - `F0` / `wav` are gathered lazily from the per-detector tables in
      `instrument_data.nc`, NaN where `detector_index < 0`.

    Constructor option `engine` (default 'netcdf4') selects the xarray I/O
    backend for every open — same rationale as the OLCI reader.
    """

    sensor = 'MERIS'
    level = 'L1'
    chunks_default = 500
    payload_var = str(names.ltoa)
    sample_products = ('MERIS_L1_SAFE_RRG', 'MERIS_L1_SAFE_FRG')
    required_vars = (str(names.sza), str(names.saa), str(names.vza), str(names.vaa),
                     str(names.F0), str(names.wav), 'quality_flags', 'detector_index')
    flags = True
    _level_token = 'Level 1'

    #: Product types of the MERIS L1 SAFE family, keyed for `detect_variant`.
    #: Detection only: `resolution` is data-driven (manifest acSpatialSampling).
    VARIANTS = {
        'RR': dict(detect=dict(field='productType', value='ME_1_RRG___')),
        'FR': dict(detect=dict(field='productType', value='ME_1_FRG___')),
    }

    def __init__(self, path, chunks=None, metadata_template=None, verbose=False,
                 engine='netcdf4'):
        # `engine` selects the xarray I/O backend for every open in this
        # reader. Default 'netcdf4': it measures substantially faster than
        # h5netcdf on these per-band .nc files (OLCI numbers) and is the
        # engine the v1 reader effectively used. Pass engine='h5netcdf' when
        # the process must stay on a single libhdf5 copy — i.e. it also uses
        # h5py-based readers or writers (SGLI, HYPSO, ECOSTRESS, PACE, GOESNG):
        # netCDF4's bundled libhdf5 poisons their dimension-scale operations
        # (REFACTORING.md, dual-libhdf5 decision).
        super().__init__(path, chunks=chunks, metadata_template=metadata_template,
                         verbose=verbose)
        self.engine = engine

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
                f'not a MERIS .SEN3 product')

        # 0. Manifest: header, level check, variant ---------------------------
        if self.verbose:
            log.debug(f'Reading MERIS {self.level} SAFE product {dirname.name}')
        manifest = read_xml(manifest_file)
        header = self._manifest_header(manifest)
        self._check_level(header['textInfo'])
        self.detect_variant(header)  # loud on an unknown productType

        # 1. Band table: manifest bandDescriptions x files on disk -------------
        labels, cwvl, files = self._band_table(manifest, dirname)

        chunks = self.chunks
        nc = {'rows': chunks[str(names.rows)], 'columns': chunks[str(names.columns)]}

        ds = xr.Dataset()

        # 2. Payload: per-band radiance rasters (lazy). `open_dataset` (not
        #    open_dataarray): each M0X file also carries a `_radiance_err` var.
        arrays = [xr.open_dataset(files[b], chunks=nc, engine=self.engine)[f'{b}_radiance']
                  .rename(_GRID_RENAME) for b in labels]
        ds[str(names.ltoa)] = xr.concat(arrays, dim=str(names.bands))
        # Radiance unit only when the files state one.
        if 'units' in arrays[0].attrs:
            ds[str(names.ltoa)].attrs['unit'] = arrays[0].attrs['units']
        # Official MERIS band names B1..B15 (N1 parity); the instrument tables
        # are re-labelled to match positionally.
        ds = ds.assign_coords({
            str(names.bands): [f'B{i + 1}' for i in range(len(labels))],
            str(names.cwav): (str(names.bands), cwvl),
            str(names.bgroup): (str(names.bands), ['bands_vnir'] * len(labels)),
        })

        # 3. Geolocation --------------------------------------------------------
        geo = xr.open_dataset(dirname / 'geo_coordinates.nc', chunks=nc,
                              engine=self.engine).rename(_GRID_RENAME)
        ac_factor = int(geo.attrs['ac_subsampling_factor'])
        al_factor = int(geo.attrs['al_subsampling_factor'])
        ds[str(names.lat)] = geo['latitude'].astype('float32')
        ds[str(names.lon)] = geo['longitude'].astype('float32')
        ds['altitude'] = geo['altitude']

        # 4. Tie-point angles (full grid, lazy blockwise interpolation) --------
        if self.verbose:
            log.debug('Read geometric tie points')
        ds = self._tie_points(ds, dirname / 'tie_geometries.nc', ac_factor, al_factor)
        ds = tiepoint_angles(ds, chunks=chunks)

        # 5. Meteorological tie points ------------------------------------------
        ds = self._meteo(ds, dirname / 'tie_meteo.nc', ac_factor, al_factor, chunks)

        # 6. Quality flags (decoded on demand via Level1_MERIS_SAFE.getflag) ----
        qf = xr.open_dataset(dirname / 'qualityFlags.nc', chunks=nc,
                             engine=self.engine).rename(_GRID_RENAME)
        ds['quality_flags'] = qf['quality_flags']

        # 7. Instrument data: per-detector tables + detector assignment --------
        ds = self._instrument_data(ds, dirname, nc)

        # 8. Per-pixel spectral quantities (wavelength, solar flux) -------------
        det = ds['detector_index']
        valid = det >= 0
        # The index is int16 with -1 invalid (kept raw); the int32 cast is for
        # the fancy indexing below, where invalids are clipped to 0 and then
        # masked back out.
        idx = det.where(valid, 0).astype('int32')
        for var, table in ((str(names.wav), 'lambda0'), (str(names.F0), 'solar_flux')):
            ds[var] = self._per_pixel(ds[table], idx).where(valid)
            ds[var].attrs.update(ds[table].attrs)

        # 9. Orientation: canonical south-up, west-to-east ----------------------
        ds = self._orient(ds)
        # The per-band payload concat leaves the bands dim one-chunk-per-band
        # while the instrument tables stay single-block; unify so every dask
        # variable reports consistent chunk boundaries (Dataset.chunks).
        ds = ds.unify_chunks()

        # 10. Attributes -----------------------------------------------------------
        filter_fn = (lambda x, y: x) if self.metadata_template is None \
            else filter_metadata
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): 'ENVISAT',
            str(names.product_name): dirname.name,
            str(names.input_directory): str(dirname.parent),
            str(names.datetime): header['datetime'],
            str(names.resolution): header['resolution'],
            str(names.shortname): header['productType'],
            str(names.crs): 'EPSG:4326',
            'user_guide': USER_GUIDE,
            'metadata': filter_fn(manifest, self.metadata_template),
        })
        ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_meris'
        ds.attrs['_srf_getter_arg'] = ''

        return self.finalize(ds)

    # ------------------------------------------------------------- manifest

    @staticmethod
    def _manifest_header(manifest: dict) -> dict:
        """Extract the identification fields used for detection and attrs."""
        objs = {o['attributes']['ID']: o
                for o in manifest['metadataSection']['metadataObject']}
        textinfo = (manifest['informationPackageMap']['contentUnit']
                    ['attributes']['textInfo'])
        gpi = objs['generalProductInformation']['metadataWrap']['xmlData'] \
            ['generalProductInformation']
        acq = objs['acquisitionPeriod']['metadataWrap']['xmlData'] \
            ['acquisitionPeriod']
        mpi = objs['merisProductInformation']['metadataWrap']['xmlData'] \
            ['merisProductInformation']

        start = datetime.fromisoformat(acq['startTime'])
        stop = datetime.fromisoformat(acq['stopTime'])

        resolution = float(mpi['samplingParameters']['acSpatialSampling'])
        if resolution.is_integer():
            resolution = int(resolution)

        return {
            'textInfo': textinfo,
            'productType': gpi['productType'],
            'datetime': (start + (stop - start) / 2.).isoformat(),
            'resolution': resolution,
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
                    if o['attributes']['ID'] == 'merisProductInformation')
        desc = (info['metadataWrap']['xmlData']['merisProductInformation']
                ['bandDescriptions'])
        labels = [k for k in desc if k != 'attributes']
        cwvl = [float(desc[k]['centralWavelength']) for k in labels]

        suffix = '_radiance.nc'
        files = {p.name[:-len(suffix)]: p
                 for p in dirname.glob(f'M*{suffix}')}
        missing = [b for b in labels if b not in files]
        extra = sorted(set(files) - set(labels))
        if missing or extra:
            raise ProductError(
                f'{self.sensor} {self.level}: band table mismatch between the '
                f'manifest bandDescriptions and the files on disk\n'
                f'  bands without file: {missing}\n'
                f'  files without band: {extra}')
        return labels, cwvl, files

    # ------------------------------------------------------------ tie points

    def _tie_points(self, ds: xr.Dataset, path: Path, ac_factor: int,
                    al_factor: int) -> xr.Dataset:
        """Open a tie-point file, align it to the grid and merge it into `ds`.

        The file's unindexed tie_rows/tie_columns dims are given pixel-position
        coordinates (arange x subsampling factor); the grid/tie alignment is
        checked loudly. (Mirrors `_OlciBase._tie_points`.)
        """
        tie = xr.open_dataset(path, engine=self.engine).rename(_ANGLE_MAP)
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
        """Interpolate the meteorological tie points to the full grid (lazy).

        Mirrors `_OlciBase._meteo`, except `humidity` is a profile over
        `tie_pressure_levels`: `spatial_resample` is 2D-only, so each level is
        resampled separately and re-stacked. The reference pressure levels are
        kept as the humidity pressure coordinate only (not as a variable).
        """
        tie = xr.open_dataset(path, engine=self.engine)
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

        # Humidity profile: resample each pressure level, re-stack (lazy)
        humidity = xr.concat(
            [spatial_resample(tie['humidity'].isel(tie_pressure_levels=i),
                              shape, tie_chunks, 'linear')
             for i in range(tie.sizes['tie_pressure_levels'])],
            dim='tie_pressure_levels',
        )
        ds['humidity'] = (humidity.rename(_TIE_GRID_RENAME)
                          .assign_coords(
                              tie_pressure_levels=
                              tie['reference_pressure_level'].values)
                          .transpose(str(names.rows), str(names.columns),
                                     'tie_pressure_levels'))
        ds['humidity'].attrs = tie['humidity'].attrs
        ds['humidity_tie'] = tie['humidity']
        return ds

    # -------------------------------------------------------- instrument data

    def _instrument_data(self, ds: xr.Dataset, dirname: Path, nc: dict) -> xr.Dataset:
        """Read the spectral tables and the per-pixel detector assignment.

        `instrument_data.nc` carries an *unindexed* bands dimension of 15
        entries (M01..M15 order); it is re-labelled to the payload band table
        so it merges without coordinate conflicts. Opened with
        `mask_and_scale=False`: the spectral tables keep their raw -1 fill in
        the band/detector slots and `detector_index` stays int16 with -1 =
        invalid (the F0/wav gather masks on it explicitly).
        """
        with warnings.catch_warnings():
            # The backend parses the raw store before drop_variables removes
            # `relative_spectral_covariance`, whose duplicated ('bands', 'bands')
            # dims trip xarray's duplicate-dimension warning at open time.
            warnings.filterwarnings('ignore',
                                    message='.*Duplicate dimension names.*')
            inst = xr.open_dataset(
                dirname / 'instrument_data.nc',
                mask_and_scale=False,
                # duplicate-dimension variable, not part of the output
                drop_variables='relative_spectral_covariance',
                chunks=nc,
                engine=self.engine,
            ).rename(_GRID_RENAME)
        nb = inst.sizes[str(names.bands)]
        if nb != 15:
            raise ProductError(
                f'{self.sensor} {self.level}: instrument_data.nc has a bands '
                f'dimension of size {nb}, expected 15')
        inst = inst.assign_coords({str(names.bands): [f'B{i + 1}' for i in range(nb)]})
        for var in ('FWHM', 'detector_index', 'frame_offset', 'lambda0', 'solar_flux'):
            ds[var] = inst[var]
        return ds

    @staticmethod
    def _per_pixel(table: xr.DataArray, detector_index: xr.DataArray) -> xr.DataArray:
        """Gather the per-detector value of a (bands, detectors) table per pixel.

        `detector_index` (rows, columns) selects one row of the table per
        pixel; apply_ufunc broadcasts the table to (bands, rows, columns,
        detectors) and the lambda picks [band, 0, 0, detector]. (Mirrors
        `_OlciBase._per_pixel`; the caller clips invalid indices to 0.)
        """
        return xr.apply_ufunc(
            lambda t, di: t[:, 0, 0, di],
            table,
            detector_index,
            dask='parallelized',
            input_core_dims=[['detectors'], []],
            output_dtypes=[table.dtype],
        )

    # ------------------------------------------------------------- orientation

    def _orient(self, ds: xr.Dataset) -> xr.Dataset:
        """Flip to the canonical south-up / west-to-east orientation, per file.

        The SAFE geolocation is NaN outside the swath (fully invalid edge rows
        plus partially filled first/last rows), so unlike the OLCI two-pixel
        check each flip is decided from the first/last *valid* latitude along
        the center column (longitude along the center row). Each check reads
        one thin line of the grid — negligible.
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

    # ----------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return ['quality_flags']

    @classmethod
    def flag_dims_like(cls) -> str:
        return 'quality_flags'

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """Bit definitions from `quality_flags.flag_meanings`.

        LAND is 'land' minus 'fresh_inland_water'; L1_INVALID is 'invalid'.
        """
        if flag_name == GenericFlags.LAND:
            land = getflag(ds['quality_flags'], 'land')
            fresh_inland_water = getflag(ds['quality_flags'], 'fresh_inland_water')
            return land & ~fresh_inland_water
        if flag_name == GenericFlags.L1_INVALID:
            return getflag(ds['quality_flags'], 'invalid')
        raise ValueError(f'Unsupported MERIS SAFE flag: {flag_name!r}')
