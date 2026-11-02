#!/usr/bin/env python3
"""
Sentinel-2 MSI (MultiSpectral Instrument) readers, levels 1C and 2A.

One class per level (REFACTORING.md, §4), sharing a sensor base:

    >>> from eoread.readers.msi import Level1C_MSI        # doctest: +SKIP
    >>> ds = Level1C_MSI('S2A_MSIL1C_...SAFE')            # doctest: +SKIP

Differences from the legacy v1 readers (documented clean breaks, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the tile (canonical
  orientation). Legacy outputs were north-up.
- L1C payload is `Ltoa` (TOA radiance) carrying a `unit` attribute; legacy
  misnamed it `Rtoa` without unit (`eoread.compat.v1` renames it back).
- L2A keeps band B8 (legacy dropped it) and, like legacy, reads each band from
  the product's own pre-downsampled file at the requested resolution when one
  exists; bands without a file at that resolution fall back to the finest
  available file, resampled onto the target grid. The band table is built from
  the files actually read — fixing the legacy mislabel of B11 as 'B10'.
- B10 is absent from baseline-5 products; missing bands are skipped with a log
  message, never dropped by name.
- `resolution=None` keeps every band on its own native grid (no resampling):
  the payloads are the stacks `Ltoa_10m` / `Ltoa_20m` / `Ltoa_60m` (`Rtoa_*m`
  for L2A), each on its own `(bands_{res}m, y_{res}m, x_{res}m)` grid with a
  per-group `cwav_{res}m` table; lat/lon and the tie-point angles live on the
  canonical `y`/`x` reference grid at 60 m. The finest (10 m) stack is the
  declared primary payload, and the `resolution` attribute carries that
  reference resolution.
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

from eoread.flags import GenericFlags
from eoread.reader import ProductError, Reader, UnsupportedProductVersion
from eoread.tools import filter_metadata, install_exit_cleanup, spatial_resample
from eoread.transforms.angles import tiepoint_angles
from eoread.transforms.geometry import latlon_utm


#: Sentinel-2 User Handbook (kept for legacy attribute parity)
USER_GUIDE = ('https://sentinels.copernicus.eu/documents/247904/685211/'
              'Sentinel-2_User_Handbook.pdf/8869acdf-fd84-43ec-ae8c-3e80a436a16c?t=1438278087000')

_PLATFORMS = {'S2A': 'Sentinel-2A', 'S2B': 'Sentinel-2B', 'S2C': 'Sentinel-2C'}
_SRF_ARGS = {'S2A': 'sentinel2_1_msi', 'S2B': 'sentinel2_2_msi', 'S2C': 'sentinel2_3_msi'}

#: Uniform-grid target resolutions (m); `resolution=None` keeps every band on
#: its own native grid (multi-resolution layout, see the module docstring).
RESOLUTIONS = (10, 20, 60)

#: Reference grid of the native layout: lat/lon and tie-point angles live on
#: the coarsest resolution (the legacy angle grid); per-payload resolutions
#: are encoded in the variable and dim names.
_NATIVE_GEOM_RES = max(RESOLUTIONS)


def _val(node):
    """Unwrap a read_xml leaf.

    Leaves without attributes parse to plain typed values; leaves carrying
    attributes come back as ``{'attributes': ..., 'values': ...}``.
    """
    if isinstance(node, dict) and 'values' in node and 'attributes' in node:
        return node['values']
    return node


def _grid(node) -> np.ndarray:
    """Parse a tie-point <Values_List> block into a float32 2D array."""
    return np.array([row.split() for row in node['Values_List']['VALUES']], dtype='float32')


def _padded_band(band: str) -> str:
    """Band label to its filename form ('B1' -> 'B01'; 'B8A', 'B10' unchanged)."""
    if band[1:].isdigit():
        return f'B{int(band[1:]):02d}'
    return band


def _discover_granule(safe: Path) -> Path:
    """Return the single granule directory of a .SAFE, or `safe` itself."""
    granule_dir = safe / 'GRANULE'
    if not granule_dir.is_dir():
        return safe
    granules = list(granule_dir.glob('*'))
    assert len(granules) == 1, f'expected exactly one granule in {safe}, got {len(granules)}'
    return granules[0]


def _viewing_angles(grids: list, shape: tuple):
    """Merge the per-(band, detector) viewing grids and return the band-0 pair.

    The detectors of a given band cover disjoint sub-regions of the tie-point
    grid (NaN elsewhere); merging by NaN-fill reconstructs the full grid. The
    merged band-0 grid is the representative viewing angle for all bands, as
    in the legacy reader.
    """
    vza, vaa = {}, {}
    for g in grids:
        bandid = int(g['attributes']['bandId'])
        for key, store in (('Zenith', vza), ('Azimuth', vaa)):
            data = _grid(g[key]).ravel()
            assert data.size == shape[0] * shape[1], \
                f'viewing angle grid of {data.size} points does not match the sun grid {shape}'
            if bandid not in store:
                store[bandid] = data.copy()
            else:
                valid = ~np.isnan(data)
                store[bandid][valid] = data[valid]
    assert 0 in vza and 0 in vaa, 'viewing angle grids do not include bandId 0'
    return vza[0].reshape(shape), vaa[0].reshape(shape)


class _MSIBase(Reader):
    """Shared implementation of the MSI level readers."""

    sensor = 'MSI'
    chunks_default = 500
    flags = True
    required_vars = (str(names.sza), str(names.saa), str(names.vza), str(names.vaa))

    # Per-level hooks, set by the subclasses ---------------------------------
    level = None
    payload_var = None
    xml_root = None
    payload_dtype = None

    def __init__(self, path, chunks=None, metadata_template=None, verbose=False, resolution=10):
        if resolution is not None and resolution not in RESOLUTIONS:
            raise ValueError(
                f'MSI: unsupported resolution {resolution!r}, expected one of (*{RESOLUTIONS}, None)')
        self.resolution = resolution
        # Native layout (resolution=None): the canonical y/x grid carries the
        # geometry at the reference resolution, and the finest (10 m) stack is
        # the declared primary payload.
        if resolution is None:
            self._geom_res = _NATIVE_GEOM_RES
            self.payload_var = f'{self.payload_var}_10m'
        else:
            self._geom_res = resolution
        super().__init__(path, chunks=chunks, metadata_template=metadata_template, verbose=verbose)

    # ------------------------------------------------------------------ read

    def read(self, path: Path) -> xr.Dataset:
        safe = Path(path).resolve()
        assert safe.exists(), f'{safe} does not exist'
        granule_dir = _discover_granule(safe)
        xmlroot_path = safe / self.xml_root
        xmlgranule_path = granule_dir / 'MTD_TL.xml'
        assert xmlroot_path.exists(), f'missing {xmlroot_path}'
        assert xmlgranule_path.exists(), f'missing {xmlgranule_path}'

        if self.verbose:
            log.debug(f'Reading MSI {self.level} product {safe.name}')

        xmlroot = read_xml(xmlroot_path)
        xmlgranule = read_xml(xmlgranule_path)

        product_image = xmlroot['General_Info']['Product_Image_Characteristics']
        baseline = float(_val(xmlroot['General_Info']['Product_Info']['PROCESSING_BASELINE']))

        # Band table, in metadata order; sensor-native labels kept verbatim
        specs = product_image['Spectral_Information_List']['Spectral_Information']
        bands = [s['attributes']['physicalBand'] for s in specs]
        cwvl = [_val(s['Wavelength']['CENTRAL']) for s in specs]
        band_res = [int(_val(s['RESOLUTION'])) for s in specs]

        tile_id = str(_val(xmlgranule['General_Info']['TILE_ID']))
        platform_code = tile_id[:3]
        assert platform_code in _PLATFORMS, f'unrecognized MSI platform {platform_code!r}'

        geocoding = xmlgranule['Geometric_Info']['Tile_Geocoding']
        size = self._geocode_node(geocoding, 'Size')
        geo = self._geocode_node(geocoding, 'Geoposition')
        ny, nx = int(size['NROWS']), int(size['NCOLS'])

        # 1. Payload
        if self.resolution is None:
            # Native layout: every band stays on its own native grid (no
            # resampling); the payloads are per-resolution stacks on separate
            # dims and the canonical y/x grid carries only the geometry.
            groups = self._read_native_bands(granule_dir, product_image, baseline, bands, band_res)
            ds = self._native_payloads(groups, bands, cwvl)
        else:
            # Read, calibrate and resample every band onto the target grid.
            arrays, kept = self._read_bands(granule_dir, product_image, baseline, bands, band_res, (ny, nx))

            ds = xr.Dataset()
            stack = da.stack([arr.data for arr in arrays])
            ds[self.payload_var] = ((str(names.bands), str(names.rows), str(names.columns)),
                                    stack.astype(self.payload_dtype))

        # 2. Geolocation (row 0 = north, as in the product)
        crs = str(_val(geocoding['HORIZONTAL_CS_CODE']))
        ll = latlon_utm(crs, float(geo['ULX']), float(geo['ULY']),
                        float(geo['XDIM']), float(geo['YDIM']), nx, ny, self.chunks)
        ds = ds.assign(ll)

        # 3. Tie-point angles interpolated onto the reference grid
        tile_angles = xmlgranule['Geometric_Info']['Tile_Angles']
        grids = tile_angles.get('Viewing_Incidence_Angles_Grids')
        if isinstance(grids, dict):
            grids = [grids]
        ds = self._read_angles(ds, tile_angles['Sun_Angles_Grid'], grids or [], (ny, nx))

        # 4. Canonical orientation: row 0 = southern edge (single vertical flip).
        # Every grid flips in one joint isel — assigning a pre-flipped variable
        # back into the dataset would re-align it against its ascending
        # coordinate and silently undo the flip (see OLI's pan grid).
        flip = {str(names.rows): slice(None, None, -1)}
        if self.resolution is None:
            for res in groups:
                flip[f'{str(names.rows)}_{res}m'] = slice(None, None, -1)
        ds = ds.isel(flip)
        coords = {str(names.rows): np.arange(ny), str(names.columns): np.arange(nx)}
        if self.resolution is None:
            for res in groups:
                rows_dim = f'{str(names.rows)}_{res}m'
                cols_dim = f'{str(names.columns)}_{res}m'
                coords[rows_dim] = np.arange(ds.sizes[rows_dim])
                coords[cols_dim] = np.arange(ds.sizes[cols_dim])
        ds = ds.assign_coords(coords)

        # 5. Spectral coordinates (uniform layout only; the native groups carry
        # their own labels and cwav tables, assigned in _native_payloads)
        if self.resolution is not None:
            ds[str(names.cwav)] = (str(names.bands), np.array([cwvl[i] for i in kept], dtype='float64'))
            ds = ds.assign_coords({str(names.bands): [bands[i] for i in kept]})
            ds[str(names.bgroup)] = (str(names.bands), [f'bands_{band_res[i]}m' for i in kept])
            ds = ds.set_coords(str(names.bgroup))

        # 6. Curated attributes (legacy parity + schema requirements)
        if self.metadata_template is None:
            metadata, metadata_granule = xmlroot, xmlgranule
        else:
            metadata = filter_metadata(xmlroot, self.metadata_template)
            metadata_granule = filter_metadata(xmlgranule, self.metadata_template)
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): _PLATFORMS[platform_code],
            str(names.product_name): safe.name,
            str(names.input_directory): str(safe.parent),
            str(names.datetime): str(_val(xmlgranule['General_Info']['SENSING_TIME'])),
            # Reference-grid resolution (uniform mode: the requested one; native
            # mode: the coarsest grid — per-payload resolutions live in the
            # variable and dim names).
            str(names.resolution): self._geom_res,
            str(names.shortname): str(_val(xmlroot['General_Info']['Product_Info']['PRODUCT_TYPE'])),
            str(names.crs): crs,
            'user_guide': USER_GUIDE,
            '_srf_getter': 'eotools.srf.get_SRF_eumetsat',
            '_srf_getter_arg': _SRF_ARGS[platform_code],
            'metadata': metadata,
            'metadata_granule': metadata_granule,
        })

        return self.finalize(ds)

    # --------------------------------------------------------------- helpers

    def _geocode_node(self, geocoding: dict, key: str) -> dict:
        """Return the Size / Geoposition entry matching the reference grid."""
        for node in geocoding.get(key) or ():
            if str(node['attributes']['resolution']) == str(self._geom_res):
                return node
        raise ProductError(f'MSI {self.level}: no {key} node for resolution {self._geom_res} m')

    def _read_bands(self, granule_dir: Path, product_image: dict, baseline: float,
                    bands: list, band_res: list, shape: tuple):
        """Read and calibrate every band with a data file onto the target grid.

        Returns (arrays, kept): dask-backed DataArrays with dims (y, x), row 0
        = tile top (north-up), and the metadata indices they correspond to.
        """
        quantif = self._quantification(product_image)
        offsets = self._offsets(product_image, baseline, len(bands))

        arrays, kept = [], []
        for i, band in enumerate(bands):
            path = self._find_band_file(granule_dir, band, band_res[i])
            if path is None:
                log.info(f'MSI {self.level}: no data file for band {band}, skipping')
                continue
            arr = self._open_calibrated(path, i, offsets, quantif)
            # Resample onto the target grid (all lazy)
            arr = spatial_resample(arr, {str(names.rows): shape[0], str(names.columns): shape[1]}, self.chunks)
            # The upsample path of spatial_resample returns the grid with swapped
            # axis labels; restore the canonical (rows, columns) order. No-op for
            # the downsample and no-change branches.
            arr = arr.transpose(str(names.rows), str(names.columns))
            arrays.append(arr)
            kept.append(i)

        assert kept, f'MSI {self.level}: no band data file found under {granule_dir}'
        install_exit_cleanup()
        return arrays, kept

    def _open_calibrated(self, path: Path, i: int, offsets: list, quantif: float):
        """Open one band file lazily and calibrate it; dims (y, x), north-up.

        Shared by the uniform-grid and native read paths.
        """
        # Open already chunked so band data stays lazy end to end: the
        # backend decodes per window at compute time and no full-resolution
        # array is ever materialized at construction. Dims are still the
        # backend's raw 'y'/'x' here; they are renamed below.
        with warnings.catch_warnings():
            # xarray's chunk normalizer warns when our chunk boundaries fall
            # inside the file's stored tiles (JP2 block layout). This is a
            # one-shot windowed read and the user-facing chunk spec takes
            # precedence over tile alignment, so the hint is pure noise here.
            warnings.filterwarnings(
                'ignore', category=UserWarning,
                message='The specified chunks separate the stored chunks')
            arr = open_dataarray(
                path, engine='rasterio',
                chunks={'y': self.chunks[str(names.rows)], 'x': self.chunks[str(names.columns)]},
            ).squeeze()
        arr = arr.rename(x=str(names.columns), y=str(names.rows))
        # Pixel-index coordinates (the upsample path of spatial_resample reads them)
        arr = arr.assign_coords({
            str(names.rows): np.arange(arr.sizes[str(names.rows)]),
            str(names.columns): np.arange(arr.sizes[str(names.columns)]),
        })
        return (arr + offsets[i]) / quantif

    def _native_band_file(self, granule_dir: Path, band: str, meta_res: int):
        """Band file for the native layout (default: the uniform-grid lookup).

        The flat L1C layout is already at native resolution, so its lookup
        applies as-is; L2A overrides to read each band's own R{res}m file.
        """
        return self._find_band_file(granule_dir, band, meta_res)

    def _read_native_bands(self, granule_dir: Path, product_image: dict, baseline: float,
                           bands: list, band_res: list) -> dict:
        """Read and calibrate every band at its own native resolution.

        No resampling: each band stays on the grid of its data file. Returns
        {res: (arrays, kept)}: per-resolution calibrated DataArrays with dims
        (y, x), row 0 = tile top (north-up), and their metadata indices.
        """
        quantif = self._quantification(product_image)
        offsets = self._offsets(product_image, baseline, len(bands))

        groups = {}
        for i, band in enumerate(bands):
            path = self._native_band_file(granule_dir, band, band_res[i])
            if path is None:
                log.info(f'MSI {self.level}: no native data file for band {band}, skipping')
                continue
            arrays, kept = groups.setdefault(band_res[i], ([], []))
            arrays.append(self._open_calibrated(path, i, offsets, quantif))
            kept.append(i)

        assert any(kept for _, kept in groups.values()), \
            f'MSI {self.level}: no band data file found under {granule_dir}'
        install_exit_cleanup()
        return groups

    def _native_payloads(self, groups: dict, bands: list, cwvl: list) -> xr.Dataset:
        """Assemble the native per-resolution payload stacks (north-up).

        Each group becomes one payload variable `{Ltoa,Rtoa}_{res}m` on its own
        (bands_{res}m, y_{res}m, x_{res}m) grid with scientific band labels
        sorted by ascending central wavelength and a per-group `cwav_{res}m`
        table. Pixel-index coordinates are assigned after the south-up flip in
        read(). The dims are relabeled from (y, x) at assignment time.
        """
        pv = type(self).payload_var  # 'Ltoa' / 'Rtoa'; the instance attr is the 10 m primary
        rows, cols = str(names.rows), str(names.columns)
        ds = xr.Dataset()
        for res in RESOLUTIONS:
            if res not in groups:
                continue
            arrays, kept = groups[res]
            # Ascending central wavelength within the group (stable; the
            # metadata order already is).
            sel = sorted(range(len(kept)), key=lambda k: cwvl[kept[k]])
            bdim = f'bands_{res}m'
            ds[f'{pv}_{res}m'] = ((bdim, f'{rows}_{res}m', f'{cols}_{res}m'),
                                  da.stack([arrays[k].data for k in sel]).astype(self.payload_dtype))
            ds[f'cwav_{res}m'] = (bdim, np.array([cwvl[kept[k]] for k in sel], dtype='float64'))
            ds[f'cwav_{res}m'].attrs['unit'] = 'nm'
            ds = ds.assign_coords({bdim: [bands[kept[k]] for k in sel]})
        return ds

    def _read_angles(self, ds: xr.Dataset, sun: dict, grids: list, shape: tuple) -> xr.Dataset:
        """Read the sun and viewing tie-point grids and interpolate them to (y, x)."""
        ny, nx = shape
        tie_vars = {
            str(names.sza): _grid(sun['Zenith']),
            str(names.saa): _grid(sun['Azimuth']),
        }
        vza_tie, vaa_tie = _viewing_angles(grids, tie_vars[str(names.sza)].shape)
        tie_vars[str(names.vza)] = vza_tie
        tie_vars[str(names.vaa)] = vaa_tie

        # Tie-point pixel positions in reference-grid space: the metadata gives
        # an explicit grid step (m); fall back to spanning the tile when absent.
        step_r = _val(sun['Zenith'].get('ROW_STEP'))
        step_c = _val(sun['Zenith'].get('COL_STEP'))
        nr, nc = tie_vars[str(names.sza)].shape
        if step_r is not None and step_c is not None:
            tie_rows = np.arange(nr) * (float(step_r) / self._geom_res)
            tie_cols = np.arange(nc) * (float(step_c) / self._geom_res)
        else:
            tie_rows = np.linspace(0, ny - 1, nr)
            tie_cols = np.linspace(0, nx - 1, nc)

        for name, data in tie_vars.items():
            ds[name + '_tie'] = (('tie_rows', 'tie_columns'), data)
        ds = ds.assign_coords(tie_rows=tie_rows, tie_columns=tie_cols)
        return tiepoint_angles(ds, chunks=self.chunks)

    # Per-level radiometry and band-file lookup (implemented by the subclasses)

    def _quantification(self, product_image: dict) -> float:
        raise NotImplementedError

    def _offsets(self, product_image: dict, baseline: float, nbands: int) -> list:
        raise NotImplementedError

    def _find_band_file(self, granule_dir: Path, band: str, meta_res: int):
        raise NotImplementedError

    # ----------------------------------------------------------------- flags

    @classmethod
    def flag_requires(cls) -> list[str]:
        return [str(names.vza)]

    @classmethod
    def flag_dims_like(cls) -> str:
        return str(names.vza)

    @classmethod
    def getflag(cls, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """No quality mask is read in v2; L1_INVALID = NaN viewing zenith angle."""
        if flag_name == GenericFlags.L1_INVALID:
            vza = ds[str(names.vza)]
            return xr.DataArray(np.isnan(vza), dims=vza.dims, coords=vza.coords)
        raise ValueError(f'Unsupported MSI flag: {flag_name!r}')


class Level1C_MSI(_MSIBase):
    """Sentinel-2 MSI Level-1C: top-of-atmosphere radiance (payload `Ltoa`)."""

    level = 'L1C'
    payload_var = str(names.ltoa)
    sample_products = ('MSI_L1C',)
    xml_root = 'MTD_MSIL1C.xml'
    # Native payload dtype: the JP2 decodes to float32 and the calibration
    # arithmetic keeps it there (NEP 50 scalar ops); legacy emitted float32.
    # Do not upcast — a float64 Ltoa doubles the full-compute peak RSS.
    payload_dtype = 'float32'

    def _quantification(self, product_image: dict) -> float:
        return float(_val(product_image['QUANTIFICATION_VALUE']))

    def _offsets(self, product_image: dict, baseline: float, nbands: int) -> list:
        # Documented deviation from the REFACTORING.md §4 example (which raised at
        # baseline >= 5.0): baseline-5 products carry the same radiometric
        # structure as 4.x, so the 4.x path applies up to baseline 6.0.
        if baseline < 4.0:
            return [0] * nbands
        if baseline >= 6.0:
            raise UnsupportedProductVersion(
                f'MSI {self.level}: unsupported processing baseline {baseline:g}')
        offsets = [int(_val(o)) for o in product_image['Radiometric_Offset_List']['RADIO_ADD_OFFSET']]
        assert len(offsets) == nbands, \
            f'MSI {self.level}: expected {nbands} radiometric offsets, got {len(offsets)}'
        return offsets

    def _find_band_file(self, granule_dir: Path, band: str, meta_res: int):
        # Flat IMG_DATA layout; every metadata band must have a file.
        img = granule_dir / 'IMG_DATA'
        pattern = f'*_{_padded_band(band)}.jp2'
        matches = list(img.glob(pattern))
        if not matches:
            raise ProductError(f'MSI L1C: no data file for band {band} in {img}')
        assert len(matches) == 1, f'ambiguous band files for {band} in {img}: {matches}'
        return matches[0]


class Level2A_MSI(_MSIBase):
    """Sentinel-2 MSI Level-2A: bottom-of-atmosphere reflectance (payload `Rtoa`)."""

    level = 'L2A'
    payload_var = str(names.rtoa)
    sample_products = ('MSI_L2A',)
    xml_root = 'MTD_MSIL2A.xml'
    # Canonical registry dtype for Rtoa.
    payload_dtype = 'float32'

    def _quantification(self, product_image: dict) -> float:
        return float(_val(product_image['QUANTIFICATION_VALUES_LIST']['BOA_QUANTIFICATION_VALUE']))

    def _offsets(self, product_image: dict, baseline: float, nbands: int) -> list:
        if baseline < 4.0:
            return [0] * nbands
        if baseline >= 6.0:
            raise UnsupportedProductVersion(
                f'MSI {self.level}: unsupported processing baseline {baseline:g}')
        offsets = [int(_val(o)) for o in product_image['BOA_ADD_OFFSET_VALUES_LIST']['BOA_ADD_OFFSET']]
        assert len(offsets) == nbands, \
            f'MSI {self.level}: expected {nbands} BOA radiometric offsets, got {len(offsets)}'
        return offsets

    def _find_band_file(self, granule_dir: Path, band: str, meta_res: int):
        """Band file for the target grid, or None.

        Prefers the product's own R{target}m file — it is the official raster
        for that grid and needs no resampling (legacy did the same); falls back
        to the finest available of IMG_DATA/R{10,20,60}m, which the caller then
        resamples onto the target grid.
        """
        img = granule_dir / 'IMG_DATA'
        pattern = f'*_{_padded_band(band)}_*.jp2'
        # Target resolution first, then finest available.
        for res in (self.resolution, *RESOLUTIONS):
            d = img / f'R{res}m'
            if not d.is_dir():
                continue
            matches = list(d.glob(pattern))
            assert len(matches) <= 1, f'ambiguous band files for {band} in {d}: {matches}'
            if matches:
                # Anomaly check: file at neither the band's native resolution
                # nor the requested one (normal pre-downsampled products match
                # one of the two and stay silent).
                if res != meta_res and res != self.resolution:
                    log.warning(f'MSI L2A: band {band} read from R{res}m, metadata says {meta_res} m')
                return matches[0]
        return None

    def _native_band_file(self, granule_dir: Path, band: str, meta_res: int):
        # Native layout: the band's own R{res}m file, no fallback and no
        # resampling (the uniform-grid lookup prefers the target grid instead).
        d = granule_dir / 'IMG_DATA' / f'R{meta_res}m'
        pattern = f'*_{_padded_band(band)}_*.jp2'
        matches = list(d.glob(pattern)) if d.is_dir() else []
        assert len(matches) <= 1, f'ambiguous band files for {band} in {d}: {matches}'
        return matches[0] if matches else None
