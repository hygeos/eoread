#!/usr/bin/env python3
"""
MERIS (Medium Resolution Imaging Spectrometer) reader, level L1 (N1 products).

    >>> from eoread.readers.meris import Level1_MERIS       # doctest: +SKIP
    >>> ds = Level1_MERIS('MER_RR__1PRACR...N1')            # doctest: +SKIP

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
from datetime import datetime
from pathlib import Path
from threading import Lock

import dask.array as da
import epr
import numpy as np
import xarray as xr

from core import log
from core.geo.naming import names

from eoread.flags import GenericFlags
from eoread.reader import Reader
from eoread.tools import filter_metadata

#: User guide for the MERIS L1 product (legacy attribute kept).
USER_GUIDE = 'https://archive.org/details/manualzilla-id-5933919/page/n13/mode/2up'

#: Per-detector characterization tables (kept in the legacy auxdata tree).
_AUXDATA_DIR = Path(__file__).resolve().parents[1] / 'legacy' / 'auxdata' / 'meris'


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

        # 1. Native 2D fields, lazily through the locked EPR API
        for name in band_names:
            if name.startswith('radiance_'):
                continue  # stacked below onto the bands dim
            band = prod.get_band(name)
            ds[name] = xr.DataArray(da.array(_EPRBand(band, lock)),
                                    dims=(rows, cols)).chunk(self.chunks)
            if band.unit is not None:
                ds[name].attrs['unit'] = band.unit
            ds[name].attrs['description'] = str(band.description)

        # 2. Ltoa: the radiance bands stacked in product order
        arrays = [da.array(_EPRBand(prod.get_band(n), lock)) for n in radiance_names]
        ltoa = xr.DataArray(da.stack(arrays), dims=(bands, rows, cols))
        ds[str(names.ltoa)] = ltoa.chunk({bands: 1, rows: self.chunks[rows],
                                          cols: self.chunks[cols]})
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
        ds = ds.assign_coords({bands: [str(i + 1) for i in range(nb_bands)]})
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
