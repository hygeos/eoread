#!/usr/bin/env python3
"""
VENµS (visible and near-infrared camera) readers, levels L1C and L2A.

One class per level (REFACTORING.md, §4), sharing a sensor base:

    >>> from eoread.readers.venus import Level1C_VENUS       # doctest: +SKIP
    >>> reader = Level1C_VENUS('VENUS-XS_..._L1C_...')       # doctest: +SKIP
    >>> ds = reader.ds                                       # doctest: +SKIP

Differences from the legacy v1 readers (documented deviations, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the granule (canonical
  orientation). Legacy outputs were north-up.
- Canonical tie-point angles (`sza`/`saa`/`vza`/`vaa`) are interpolated onto
  the full grid from the metadata 2x2 tie grids, which are kept as `*_tie`
  variables; the legacy reader computed no angles at all. Viewing angles are
  the mean of the four detector grids.
- `SOL_ALL` / `VIE_ALL` are kept raw on their own native tie grids (separate
  dim names), so VIE is no longer silently reindexed onto SOL's smaller grid.
- L2A surface reflectance: pixels holding the product nodata value (-10000)
  become NaN (legacy kept them as -10.0 after calibration).
- L2A `water_vapor` / `aod`: ATB counts are calibrated with the MTD
  quantification factors (legacy kept raw counts).
- L2A `PIX_XS`: assigned positionally onto the bands dim; the legacy int band
  index vs string coordinate alignment produced an all-NaN variable.
- lat/lon: proper per-block UTM transform via `latlon_utm`; the legacy crude
  corner linear interpolation was flipped relative to the raster.
- `crs` and `shortname` attributes are now set (legacy did not).
"""

import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from xarray import open_dataarray

from core import log
from core.geo.naming import names
from core.table import read_xml

from eoread.flags import FlagsReaderBase, GenericFlags
from eoread.reader import ProductError, Reader, UnsupportedProductType
from eoread.tools import (filter_metadata, install_exit_cleanup,
                          spatial_resample)
from eoread.transforms.angles import tiepoint_angles
from eoread.transforms.geometry import latlon_utm

#: VENµS product format user guide (legacy attribute parity).
USER_GUIDE = 'https://www.cesbio.cnrs.fr/multitemp/ven%c2%b5s-product-format/'

_SRF_URL = ('https://labo.obs-mip.fr/wp-content-labo/uploads/sites/19/2018/09/rep6S.txt')


def get_SRF(
        ds_in: xr.Dataset | None = None,
        dir_data: Path | None = None
) -> xr.Dataset:
    """
    Load VENµS spectral response functions (SRF) for radiometric calculations.

    Downloads SRF data from the official repository if not already cached.

    Parameters
    ----------
    ds_in : xr.Dataset, optional
        Optional dataset with band names. If provided, output bands
        are referenced by ds_in.bands. Otherwise uses band IDs 1-12.
        Default is None.
    dir_data : Path, optional
        Directory to cache SRF data. If None, uses default static directory.
        Default is None.

    Returns
    -------
    xr.Dataset
        Dataset containing:
            - SRF curves for each VENµS band
            - wav: Wavelength coordinate in nanometers
            - Band variables named by band ID or from ds_in.bands
    """
    # Local imports keep this module importable in environments without the
    # static-data/network stack (the smoke gate imports every reader module).
    from core import env
    from core.files import mdir
    from core.network.download import download_url
    from core.table import read_csv

    if dir_data is None:
        dir_data = mdir(env.getdir('DIR_STATIC') / 'venus')

    srf_file = download_url(_SRF_URL, dir_data)
    nbands = 12
    ibands = range(1, nbands + 1)
    df = read_csv(srf_file, sep=None, names=['wav_um', *ibands])

    ds = xr.Dataset()
    ds.attrs['desc'] = 'Spectral response functions for VENµS'

    if ds_in is None:
        bids = ibands
    else:
        assert len(ds_in.bands) == nbands
        bids = ds_in.bands.values
    for i in range(nbands):
        ds[bids[i]] = xr.DataArray(
            df[ibands[i]].values,
            dims=['wav'],
            attrs={'band_info': f'VENUS band {bids[i]}'},
        )

    ds = ds.assign_coords(wav=df['wav_um'].values * 1000)
    ds[str(names.wav)].attrs['units'] = 'nm'

    return ds


def _val(node):
    """Unwrap a `read_xml` leaf.

    A leaf without attributes is parsed as a plain typed value; a leaf with
    attributes comes back as ``{'attributes': ..., 'values': ...}``.
    """
    if isinstance(node, dict) and 'values' in node and 'attributes' in node:
        return node['values']
    return node


def _find_file(dirname: Path, pattern: str) -> Path:
    """Return the unique file matching `pattern` under `dirname`.

    The pattern must include the extension (pathlib quirk: `*CLM_XS` does not
    match `..._CLM_XS.tif`).
    """
    matches = sorted(dirname.glob(pattern))
    assert len(matches) == 1, f'expected exactly one {pattern!r} in {dirname}, got {len(matches)}'
    return matches[0]


def _grid_node(node):
    """Return the granule's `group_id='XS'` Group_Geopositioning entry.

    `read_xml` returns a singleton child element as a dict and repeated ones
    as a list; normalize both cases and select by the group id attribute.
    """
    entries = [node] if isinstance(node, dict) else list(node)
    for entry in entries:
        attrs = entry.get('attributes') or {}
        if str(attrs.get('group_id', '')) == 'XS':
            return entry
    raise ProductError(f'no group_id=XS Group_Geopositioning entry found (saw {len(entries)})')


def _grid64(node) -> np.ndarray:
    """Parse a tie-point `Values_List` block into a float64 2D array."""
    return np.array([row.split() for row in node['Values_List']['VALUES']], dtype='float64')


def _zip_raster_path(path: Path) -> str:
    """Return the GDAL ``/vsizip`` path of the single raster in a zip archive.

    Keeping GDAL on the archive member makes the mask reads lazy: no temporary
    extraction is needed, so the data source outlives the reader call and dask
    can fetch each window when it is actually computed.
    """
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith('/')]
    if len(members) != 1:
        raise ProductError(
            f'expected exactly one raster in {path.name}, found {len(members)}: {members}')
    return f'/vsizip/{path}/{members[0]}'


class _VENUSBase(Reader):
    """Shared implementation of the VENUS level readers."""

    sensor = 'VENUS'
    chunks_default = 500
    required_vars = (str(names.sza), str(names.saa), str(names.vza), str(names.vaa))
    srf_getter = None

    # Per-level hooks, set by the subclasses ---------------------------------
    level = None
    payload_var = None
    flags = None

    # ------------------------------------------------------------------ read

    def read(self, path: Path) -> xr.Dataset:
        granule_dir = Path(path).resolve()
        assert granule_dir.exists(), f'{granule_dir} does not exist'

        if self.verbose:
            log.debug(f'Reading VENUS {self.level} product {granule_dir.name}')

        mtd_path = _find_file(granule_dir, '*MTD_ALL.xml')
        uii_path = _find_file(granule_dir / 'DATA', '*UII_ALL.xml')
        mtd = read_xml(mtd_path)
        uii = read_xml(uii_path)

        product_char = mtd['Product_Characteristics']
        level = str(_val(product_char['PRODUCT_LEVEL']))
        platform = str(_val(product_char['PLATFORM']))
        if level != self.level or platform != 'VENUS':
            raise UnsupportedProductType(
                f'expected a VENUS {self.level} product, got {platform} {level}')

        # Band table, in metadata order; sensor-native labels kept verbatim
        radio = mtd['Radiometric_Informations']
        specs = radio['Spectral_Band_Informations_List']['Spectral_Band_Informations']
        bands = [str(_val(s['attributes']['band_id'])) for s in specs]
        cwvl = [float(_val(s['Wavelength']['CENTRAL'])) for s in specs]
        band_res = {int(_val(s['SPATIAL_RESOLUTION'])) for s in specs}
        assert len(band_res) == 1, f'inconsistent band resolutions in the MTD: {band_res}'
        self.resolution = band_res.pop()

        quantif = float(_val(radio['REFLECTANCE_QUANTIFICATION_VALUE']))

        # Geopositioning: granule grid and CRS
        geo = mtd['Geoposition_Informations']
        grid = _grid_node(geo['Geopositioning']['Group_Geopositioning_List']['Group_Geopositioning'])
        ulx, uly = float(grid['ULX']), float(grid['ULY'])
        xdim, ydim = float(grid['XDIM']), float(grid['YDIM'])
        ny, nx = int(grid['NROWS']), int(grid['NCOLS'])

        code = str(_val(geo['Coordinate_Reference_System']['Horizontal_Coordinate_System']
                        ['HORIZONTAL_CS_CODE']))
        crs = code if code.startswith('EPSG') else f'EPSG:{code}'

        ds = xr.Dataset()

        # 1. Level-specific variables (payload, masks); all full-grid vars are
        #    assigned as raw data with explicit dims so no UTM indexes ever
        #    leak onto the granule grid.
        self._read_level_vars(ds, granule_dir, radio, bands, quantif, ny, nx)
        install_exit_cleanup()

        # 2. Geolocation (row 0 = north, as in the product)
        ll = latlon_utm(crs, ulx, uly, xdim, ydim, nx, ny, self.chunks)
        ds = ds.assign(ll)

        # 3. Tie-point angles interpolated onto the full grid
        ds = self._read_angles(ds, mtd, ny, nx, xdim, ydim)

        # 4. Angle rasters (SOL then VIE), kept raw on their native tie grids
        ds = self._read_angle_rasters(ds, granule_dir / 'DATA')

        # 5. Canonical orientation: row 0 = southern edge (single vertical flip)
        ds = ds.isel(y=slice(None, None, -1))
        ds = ds.assign_coords({
            str(names.rows): np.arange(ny),
            str(names.columns): np.arange(nx),
        })

        # 6. Spectral coordinates
        ds[str(names.cwav)] = (str(names.bands), np.array(cwvl, dtype='float64'))
        ds = ds.assign_coords({str(names.bands): bands})
        ds[str(names.bgroup)] = (str(names.bands), ['bands_vnir'] * len(bands))
        ds = ds.set_coords(str(names.bgroup))

        # 7. Curated attributes (legacy parity + schema requirements); the
        #    metadata copies are filtered, the parsed dicts stay intact.
        if self.metadata_template is None:
            metadata, mtd_attr = uii, mtd
        else:
            metadata = filter_metadata(uii, self.metadata_template)
            mtd_attr = filter_metadata(mtd, self.metadata_template)
        ds.attrs.update({
            str(names.sensor): self.sensor,
            str(names.platform): platform,
            str(names.product_name): str(_val(product_char['PRODUCT_ID'])),
            str(names.input_directory): str(granule_dir.parent),
            str(names.datetime): str(
                datetime.fromisoformat(str(_val(product_char['ACQUISITION_DATE'])))),
            str(names.resolution): self.resolution,
            str(names.shortname): str(_val(product_char['PRODUCT_ID'])),
            str(names.crs): crs,
            'user_guide': USER_GUIDE,
            'metadata': metadata,
            'metadata_granule': mtd_attr,
        })
        if self.srf_getter is not None:
            ds.attrs['_srf_getter'] = self.srf_getter
            ds.attrs['_srf_getter_arg'] = None

        return ds

    # --------------------------------------------------------------- helpers

    def _read_level_vars(self, ds: xr.Dataset, dirname: Path, radio: dict, bands: list,
                         quantif: float, ny: int, nx: int) -> None:
        """Read the level-specific payload and masks onto `ds`. Hook."""
        raise NotImplementedError

    def _lazy_open(self, path: Path | str) -> xr.DataArray:
        """Open a raster lazily with the reader chunk spec; drop the CRS coord.

        ``path`` may be a regular file or a GDAL ``/vsizip`` pseudo-path (see
        `_zip_raster_path`).  xarray's rasterio backend rejects pseudo-paths
        because it checks for a real file first, so the archive branch goes
        through rioxarray, which passes the path straight to GDAL.
        """
        path = str(path)
        chunks = {'y': self.chunks[str(names.rows)],
                  'x': self.chunks[str(names.columns)]}
        with warnings.catch_warnings():
            # xarray's chunk normalizer warns when our chunk boundaries fall
            # inside the file's stored tiles (one-shot windowed read; the
            # user-facing chunk spec takes precedence, so the hint is noise).
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message='The specified chunks separate the stored chunks')
            if path.startswith('/vsizip/'):
                import rioxarray
                arr = rioxarray.open_rasterio(
                    path, chunks=chunks, masked=True, mask_and_scale=True,
                ).squeeze()
            else:
                arr = open_dataarray(
                    path, engine='rasterio', chunks=chunks,
                ).squeeze()
        if 'spatial_ref' in arr.coords:
            arr = arr.drop_vars('spatial_ref')
        return arr

    def _read_angles(self, ds: xr.Dataset, mtd: dict, ny: int, nx: int,
                     xdim: float, ydim: float) -> xr.Dataset:
        """Read the sun and viewing tie grids and interpolate them onto (y, x)."""
        geo = mtd['Geometric_Informations']['Angles_Grids_List']
        sun = geo['Sun_Angles_Grids']

        # The metadata states the tie-point grid step in metres; sanity-check it
        # against the granule grid before trusting it.
        step_r = float(_val(sun['Zenith']['ROW_STEP']))
        step_c = float(_val(sun['Zenith']['COL_STEP']))
        expect_r = (ny - 1) * abs(ydim)
        expect_c = (nx - 1) * abs(xdim)
        assert abs(step_r - expect_r) <= 1e-3 * expect_r, \
            f'Venus sun grid ROW_STEP {step_r} does not span the granule (expected ~{expect_r})'
        assert abs(step_c - expect_c) <= 1e-3 * expect_c, \
            f'Venus sun grid COL_STEP {step_c} does not span the granule (expected ~{expect_c})'

        # Tie-point pixel positions: full extent, two points per axis.
        tie_rows = np.linspace(0, ny - 1, 2)
        tie_cols = np.linspace(0, nx - 1, 2)

        vza_tie, vaa_tie = self._viewing_angles(geo)
        tie_vars = {
            str(names.sza): _grid64(sun['Zenith']).astype('float32'),
            str(names.saa): _grid64(sun['Azimuth']).astype('float32'),
            str(names.vza): vza_tie,
            str(names.vaa): vaa_tie,
        }
        for name, data in tie_vars.items():
            ds[name + '_tie'] = (('tie_rows', 'tie_columns'), data)
        ds = ds.assign_coords(tie_rows=tie_rows, tie_columns=tie_cols)
        return tiepoint_angles(ds, chunks=self.chunks)

    @staticmethod
    def _viewing_angles(geo: dict) -> tuple[np.ndarray, np.ndarray]:
        """Merge the per-detector viewing tie grids by float64 averaging."""
        grids = geo['Viewing_Incidence_Angles_Grids_List'][
            'Band_Viewing_Incidence_Angles_Grids_List']['Viewing_Incidence_Angles_Grids']
        if isinstance(grids, dict):
            grids = [grids]
        vza, vaa = [], []
        for g in grids:
            z, a = _grid64(g['Zenith']), _grid64(g['Azimuth'])
            assert z.shape == (2, 2) and a.shape == (2, 2), \
                f'Venus viewing angle grid has unexpected shape {z.shape}'
            vza.append(z)
            vaa.append(a)
        return (np.mean(vza, axis=0)).astype('float32'), (np.mean(vaa, axis=0)).astype('float32')

    def _read_angle_rasters(self, ds: xr.Dataset, data_dir: Path) -> xr.Dataset:
        """Read SOL_ALL / VIE_ALL and keep them raw on their native tie grids.

        The two rasters carry different band/tie grids; each keeps its own dim
        names (bands_angle/y_tie/x_tie vs bands_view/y_view/x_view) together
        with its native tie UTM coordinates, so xarray never reindexes one onto
        the other's coordinates. Both dim sets are new to the dataset, so
        assigning the DataArrays directly is safe; the granule grid itself stays
        unindexed (full-grid variables are assigned as raw data).
        """
        sol = self._lazy_open(_find_file(data_dir, '*SOL_ALL.tif'))
        ds['SOL_ALL'] = sol.rename(band='bands_angle', x='x_tie', y='y_tie')

        vie = self._lazy_open(_find_file(data_dir, '*VIE_ALL.tif'))
        ds['VIE_ALL'] = vie.rename(band='bands_view', x='x_view', y='y_view')
        return ds


class Level1C_VENUS(_VENUSBase):
    """VENµS level L1C: top-of-atmosphere reflectance (payload `Rtoa`)."""

    level = 'L1C'
    payload_var = str(names.rtoa)
    flags = 'eoread.readers.venus.FlagsReader_VENUS'
    srf_getter = 'eoread.readers.venus.get_SRF'

    def _read_level_vars(self, ds, dirname, radio, bands, quantif, ny, nx):
        # TOA reflectance: one lazy open per band file, calibrated in float32
        arrays = []
        for band in bands:
            arr = self._lazy_open(_find_file(dirname, f'*REF_{band}.tif')).squeeze()
            assert tuple(arr.shape) == (ny, nx), \
                f'Venus L1C REF_{band} shape {tuple(arr.shape)} does not match the MTD grid ({ny}, {nx})'
            arrays.append((arr / quantif).astype('float32').data)

        ds[self.payload_var] = ((str(names.bands), str(names.rows), str(names.columns)),
                                da.stack(arrays))
        ds[self.payload_var].attrs['unit'] = None

        # Classification map: coarse native grid, resampled onto the pixel grid
        cld = self._lazy_open(_find_file(dirname / 'DATA', '*CLA_ALL.tif'))
        cld = spatial_resample(cld, {str(names.rows): ny, str(names.columns): nx},
                               self.chunks, 'repeat')
        cld = cld.transpose(str(names.rows), str(names.columns))
        ds['CLA_ALL'] = (cld.dims, cld.data)

        # Full-grid quality masks. The zipped rasters are opened through GDAL's
        # /vsizip path: no temporary extraction, no eager materialization, and
        # the archive stays readable for every lazily computed window.
        for var in ('CLD_XS', 'USI_XS'):
            path = _find_file(dirname / 'MASKS', f'*{var}.zip')
            arr = self._lazy_open(_zip_raster_path(path))
            assert tuple(arr.shape) == (ny, nx), \
                f'Venus L1C {var} shape {tuple(arr.shape)} does not match the MTD grid ({ny}, {nx})'
            ds[var] = ((str(names.rows), str(names.columns)), arr.data)

        # Per-band quality masks merged onto the bands dim (positional, MTD order)
        for name, var in (('PIX', 'PIX_XS'), ('SAT', 'SAT_XS')):
            arrays = []
            for band in bands:
                path = _find_file(dirname / 'MASKS', f'*{name}_{band}.zip')
                arr = self._lazy_open(_zip_raster_path(path))
                assert tuple(arr.shape) == (ny, nx), \
                    f'Venus L1C {name}_{band} shape {tuple(arr.shape)} does not match the MTD grid ({ny}, {nx})'
                arrays.append(arr.data)
            ds[var] = ((str(names.bands), str(names.rows), str(names.columns)), da.stack(arrays))


class Level2A_VENUS(_VENUSBase):
    """VENµS level L2A: surface reflectance (payloads `rho_surface` / `rho_flat`)."""

    level = 'L2A'
    payload_var = 'rho_surface'
    flags = 'eoread.readers.venus.FlagsReader_Level2A'

    def _read_level_vars(self, ds, dirname, radio, bands, quantif, ny, nx):
        # Product nodata raw value (SVL 'nodata' entry; absent on L1C)
        svl = radio.get('Special_Values_List')
        nodata_node = svl.get('nodata') if isinstance(svl, dict) else None
        nodata = float(nodata_node['values']) \
            if isinstance(nodata_node, dict) and 'values' in nodata_node else None

        # Surface reflectances: calibrate first (float32), then mask nodata
        for var, prefix in (('rho_surface', 'SRE'), ('rho_flat', 'FRE')):
            arrays = []
            for band in bands:
                arr = self._lazy_open(_find_file(dirname, f'*{prefix}_{band}.tif')).squeeze()
                assert tuple(arr.shape) == (ny, nx), \
                    f'Venus L2A {prefix}_{band} shape {tuple(arr.shape)} does not match the MTD grid ({ny}, {nx})'
                calib = (arr / quantif).astype('float32')
                if nodata is not None:
                    calib = calib.where(arr != nodata)
                arrays.append(calib.data)
            ds[var] = ((str(names.bands), str(names.rows), str(names.columns)), da.stack(arrays))
            ds[var].attrs['unit'] = None

        # Atmospheric state band: raw counts calibrated with the MTD quantifications
        atb = self._lazy_open(_find_file(dirname, '*ATB_XS.tif'))
        assert tuple(atb.shape) == (2, ny, nx), \
            f'Venus L2A ATB shape {tuple(atb.shape)} does not match ((2,), {ny}, {nx})'
        q_wv = float(_val(radio['WATER_VAPOR_CONTENT_QUANTIFICATION_VALUE']))
        q_aot = float(_val(radio['AEROSOL_OPTICAL_THICKNESS_QUANTIFICATION_VALUE']))
        ds['water_vapor'] = ((str(names.rows), str(names.columns)),
                             (atb.isel(band=0) / q_wv).astype('float32').data)
        ds['aod'] = ((str(names.rows), str(names.columns)),
                     (atb.isel(band=1) / q_aot).astype('float32').data)

        # Full-grid quality masks
        for var in ('CLM_XS', 'USI_XS', 'SAT_XS', 'IAB_XS', 'EDG_XS'):
            arr = self._lazy_open(_find_file(dirname / 'MASKS', f'*{var}.tif')).squeeze()
            assert tuple(arr.shape) == (ny, nx), \
                f'Venus L2A {var} shape {tuple(arr.shape)} does not match the MTD grid ({ny}, {nx})'
            ds[var] = ((str(names.rows), str(names.columns)), arr.data)

        # Per-band pixel mask: assigned positionally onto the unindexed bands
        # dim — the file's integer band index is discarded (the legacy int band
        # index vs string coordinate alignment silently produced all NaN).
        pix = self._lazy_open(_find_file(dirname / 'MASKS', '*PIX_XS.tif'))
        assert tuple(pix.shape) == (len(bands), ny, nx), \
            f'Venus L2A PIX shape {tuple(pix.shape)} does not match ({len(bands)}, {ny}, {nx})'
        ds['PIX_XS'] = ((str(names.bands), str(names.rows), str(names.columns)), pix.data)


class FlagsReader_VENUS(FlagsReaderBase):
    """VENµS L1C flags (legacy semantics kept verbatim)."""

    def requires(self) -> list[str]:
        return ['CLA_ALL', 'CLD_XS']

    def dims_like(self) -> str:
        return 'CLA_ALL'

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.L1_INVALID:
            cla = ds['CLA_ALL']
            return xr.DataArray(cla.isnull(), dims=cla.dims, coords=cla.coords)
        elif flag_name == GenericFlags.CLOUD:
            cld = ds['CLD_XS']
            return xr.DataArray(cld, dims=cld.dims, coords=cld.coords)
        else:
            raise ValueError(f'Unsupported flag: {flag_name}')


class FlagsReader_Level2A(FlagsReaderBase):
    """VENµS L2A flags (new in v2; the legacy L2A reader set no flag reader)."""

    def requires(self) -> list[str]:
        return ['rho_surface', 'CLM_XS']

    def dims_like(self) -> str:
        return 'CLM_XS'

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        if flag_name == GenericFlags.L1_INVALID:
            rho = ds['rho_surface']
            return rho.isnull().any(dim=str(names.bands))
        elif flag_name == GenericFlags.CLOUD:
            return ds['CLM_XS'] > 0
        else:
            raise ValueError(f'Unsupported flag: {flag_name}')
