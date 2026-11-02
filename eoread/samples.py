#!/usr/bin/env python3
"""
Sample-product registry.

Reader classes only declare *which* sample products they support via their
``sample_products`` attribute.  This module owns the out-of-reader details:
the actual location of each sample product.

Samples are declared as explicit paths (relative to the repository for the
bundled products, absolute for the shared Ceph archive).  A missing local
path raises a loud, actionable error — except when the sample carries a
:class:`Download` spec: it is then fetched on demand through SAND into the
local cache (``~/.cache/eoread/samples``, override with ``EOREAD_SAMPLES_DIR``)
and resolved from there.  The ``sand`` package (``samples`` extra) is
imported lazily, only when a download is actually attempted, so this module
stays importable without it.

Making a sensor downloadable is one line per product: register the sample
with ``download=Download(provider, provider_product_id)``.  The provider must
be one of ``cdse``/``cnes``/``eumdac``/``nasa``/``usgs``; provider
credentials (when required) live in ``~/.netrc``.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from core.import_utils import import_module


#: Shared sample archive (developer machines).
_CEPH = Path('/mnt/ceph/samples')

#: Repository root — bundled products are declared relative to it.
_REPO = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Download:
    """How to fetch a missing sample from its provider through SAND.

    Parameters
    ----------
    provider : str
        SAND provider key: 'cdse', 'cnes', 'eumdac', 'nasa' or 'usgs'.
    product_id : str
        Exact product identifier as indexed by the provider (the same value
        passed to ``sand download`` on the command line).
    """

    provider: str
    product_id: str


@dataclass(frozen=True)
class Sample:
    """A sample product available for one (sensor, level, product type).

    Parameters
    ----------
    path : Path
        Location of the sample file or product directory.
    description : str, optional
        Human-readable description used in listings and error messages.
    download : Download, optional
        Fallback used when ``path`` is missing: fetch the sample through SAND
        into the local cache (see :func:`_download_sample`).  Samples without
        a spec simply raise when their path is missing.
    """

    path: Path
    description: str = ''
    download: Download | None = None

    def resolve(self) -> Path:
        """Return the sample path, or raise ``FileNotFoundError``.

        Relative paths are anchored to the repository root (bundled products
        under ``data/sample_products/``); absolute paths are used as-is.

        When the local path is missing and a :class:`Download` spec is set,
        the sample is fetched through SAND into the local cache instead of
        raising.
        """
        path = Path(self.path).expanduser()
        if not path.is_absolute():
            path = _REPO / path
        if path.exists():
            return path
        if self.download is None:
            raise FileNotFoundError(
                f'{self.description or "sample"}: sample path does not exist: {path}')
        return _download_sample(self)


#: SAND downloader classes by provider key (mirrors the `sand download` CLI).
_PROVIDERS = {
    'cdse': 'sand.copernicus_dataspace.DownloadCDSE',
    'cnes': 'sand.cnes.DownloadCNES',
    'eumdac': 'sand.eumdac.DownloadEumDAC',
    'nasa': 'sand.nasa.DownloadNASA',
    'usgs': 'sand.usgs.DownloadUSGS',
}


def _cache_dir() -> Path:
    """Root directory for provider-downloaded samples.

    Defaults to ``~/.cache/eoread/samples``; override with the
    ``EOREAD_SAMPLES_DIR`` environment variable.
    """
    override = os.environ.get('EOREAD_SAMPLES_DIR')
    if override:
        return Path(override).expanduser()
    return Path.home() / '.cache' / 'eoread' / 'samples'


def _make_downloader(provider: str):
    """Instantiate the SAND downloader for `provider`.

    Raises ``FileNotFoundError`` (never ``ImportError``) when the provider is
    unknown or the ``sand`` package (``samples`` extra) is not installed.
    """
    module_path = _PROVIDERS.get(provider)
    if module_path is None:
        known = ', '.join(sorted(_PROVIDERS))
        raise FileNotFoundError(
            f'unknown SAND provider {provider!r} (known: {known})')
    try:
        return import_module(module_path)(verbose=False)
    except ImportError as exc:
        raise FileNotFoundError(
            f'the "sand" package is not available ({exc}); install the '
            f'"samples" extra to enable sample downloads') from None


def _download_sample(sample: Sample) -> Path:
    """Fetch a missing sample through SAND into the local cache.

    Every failure mode raises ``FileNotFoundError`` so callers (tests,
    scripts) degrade to "sample unavailable" — a clean skip — instead of an
    error cascade.
    """
    spec = sample.download
    label = sample.description or 'sample'
    try:
        downloader = _make_downloader(spec.provider)  # may raise FileNotFoundError
    except FileNotFoundError as exc:
        raise type(exc)(f'{label}: {exc}') from None
    try:
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        target = downloader.download_file(spec.product_id, cache)
    except Exception as exc:  # noqa: BLE001 — any provider failure means "no sample"
        raise FileNotFoundError(
            f'{label}: SAND download of {spec.product_id} failed: {exc}') from None
    target = Path(target)
    if not target.exists():
        raise FileNotFoundError(f'{label}: SAND reported a missing target: {target}')
    return target


#: Registry keyed by stable sample-product IDs.  Reader classes reference these
#: IDs from their ``sample_products`` attribute; they never contain paths.
_samples: dict[str, Sample] = {
    # Sentinel-2 MSI (downloadable: cdse)
    # The downloadable L1C is a different granule (S2A, UTM zone 23) from the
    # Ceph copy (S2B, UTM zone 31); tests must stay granule-agnostic.  The L2A
    # download is the same granule as the Ceph copy.
    'MSI_L1C': Sample(
        _CEPH / 'SENTINEL-2-MSI' / 'S2B_MSIL1C_20250320T104639_N0511_R051_T31UDS_20250320T142408.SAFE',
        'Sentinel-2 MSI L1C',
        download=Download('cdse', 'S2A_MSIL1C_20230617T130251_N0510_R095_T23KPQ_20240905T221745.SAFE')),
    'MSI_L2A': Sample(
        _CEPH / 'SENTINEL-2-MSI' / 'S2B_MSIL2A_20250930T102739_N0511_R108_T31TFJ_20250930T142814.SAFE',
        'Sentinel-2 MSI L2A',
        download=Download('cdse', 'S2B_MSIL2A_20250930T102739_N0511_R108_T31TFJ_20250930T142814.SAFE')),

    # Sentinel-3 OLCI
    # The Ceph copy of the 2016 EFR granule is not traversable (missing
    # execute bit, owner francois); the bundled 2025 granule below is used.
    'OLCI_L1_EFR': Sample(
        _CEPH / 'SENTINEL-3-OLCI' / 'S3A_OL_1_EFR____20250101T091103_20250101T091403_20250102T113753_0180_121_050_2700_MAR_O_NT_004.SEN3',
        'Sentinel-3 OLCI L1 EFR'),
    'OLCI_L1_ERR': Sample(
        _CEPH / 'SENTINEL-3-OLCI' / 'S3A_OL_1_ERR____20230314T234741_20230315T003137_20230316T043356_2636_096_287______MAR_O_NT_002.SEN3',
        'Sentinel-3 OLCI L1 ERR (reduced resolution)'),
    'OLCI_L2_WFR': Sample(
        _CEPH / 'SENTINEL-3-OLCI' / 'S3A_OL_2_WFR____20250930T063451_20250930T063751_20251001T125533_0180_131_077_2520_MAR_O_NT_003.SEN3',
        'Sentinel-3 OLCI L2 WFR'),
    'OLCI_L2_WRR': Sample(
        _CEPH / 'SENTINEL-3-OLCI' / 'S3A_OL_2_WRR____20250929T233650_20250930T002104_20251001T022537_2654_131_073______MAR_O_NT_003.SEN3',
        'Sentinel-3 OLCI L2 WRR'),

    # MODIS
    'MODIS_L1_AQUA': Sample(
        _CEPH / 'MODIS_AQUA' / 'MYD021KM.A2018238.1740.006.2018239145651.hdf',
        'Aqua MODIS L1B'),

    # Landsat OLI
    'OLI_L1TP_L8': Sample(
        _CEPH / 'LANDSAT_OLI' / 'LC08_L1TP_044034_20250620_20250627_02_T1',
        'Landsat-8 OLI L1TP'),
    'OLI_L1TP_L9': Sample(
        _CEPH / 'LANDSAT_OLI' / 'LC09_L1TP_223084_20250618_20250618_02_T1',
        'Landsat-9 OLI L1TP'),

    # ECOSTRESS (L1C downloadable: nasa; the v001 L1B/L2 maquette products have
    # no SAND catalog entry and stay Ceph-only)
    # The downloadable L1C is a different granule (tile 30110_005) from the
    # Ceph copy (tile 28710_008, night pass); tests must stay granule-agnostic.
    'ECOSTRESS_L1C': Sample(
        _CEPH / 'ECOSTRESS' / 'ECOv002_L1CG_RAD_28710_008_20230730T032333_0710_01.h5',
        'ECOSTRESS L1C radiance',
        download=Download('nasa', 'ECOv002_L1CG_RAD_30110_005_20231028T094350_0711_01.h5')),
    'ECOSTRESS_L1B': Sample(
        _CEPH / 'ECOSTRESS' / 'ECOSTRESS_L1B_RAD_03753_010_20190305T052421_0601_02.h5',
        'ECOSTRESS L1B radiance'),
    'ECOSTRESS_L2_V001': Sample(
        _CEPH / 'ECOSTRESS' / 'ECOSTRESS_L2_LSTE_03753_010_20190305T052421_0601_02.h5',
        'ECOSTRESS L2 LSTE (v001)'),
    'ECOSTRESS_L2_V002': Sample(
        _CEPH / 'ECOSTRESS' / 'ECOv002_L2_LSTE_37277_002_20250201T125324_0713_01.h5',
        'ECOSTRESS L2 LSTE (v002)'),

    # MERIS
    'MERIS_L1': Sample(
        _CEPH / 'MERIS' /
        'MER_RR__1PRACR20080701_014028_000026402070_00003_33123_0000.N1',
        'MERIS L1'),

    # VENµS (downloadable: cnes)
    # The downloadable granule is the BOMBETOK tile (Madagascar, EPSG:32738),
    # not the Ceph VILAINE tile (France, EPSG:32630); tests must derive the
    # expected CRS from the product itself, never pin a zone.
    # NOTE: known CNES STAC API issue — identifier "contains" is fuzzy token
    # search, so the exact-ID query matches ~49 granules and SAND's
    # uniqueness check fails; these downloads degrade to a clean skip until
    # fixed upstream (the IDs above are correct).
    'VENUS_L1C': Sample(
        _CEPH / 'VENUS' / 'VENUS-XS_20230116-112657-000_L1C_VILAINE_C_V3-1',
        'VENµS L1C',
        download=Download('cnes', 'VENUS-XS_20240718-055823-000_L1C_BOMBETOK_D')),
    'VENUS_L2A': Sample(
        _CEPH / 'VENUS' / 'VENUS-XS_20230116-112657-000_L2A_VILAINE_C_V3-1',
        'VENµS L2A',
        download=Download('cnes', 'VENUS-XS_20240718-055823-000_L2A_BOMBETOK_D')),

    # PACE OCI
    'PACE_L1B': Sample(
        _CEPH / 'PACE_OCI' / 'PACE_OCI.20250430T130447.L1B.V3.nc',
        'PACE OCI L1B'),

    # GCOM-C SGLI
    'SGLI_L1B': Sample(
        _CEPH / 'SGLI' / 'GC1SG1_201912050159F05712_1BSG_VNRDQ_1007.h5',
        'GCOM-C SGLI L1B'),

    # ISS HICO (OB.DAAC L1B_ISS tiles)
    'HICO_L1B': Sample(
        _CEPH / 'HICO' / 'H2014069115230.L1B_ISS.nc',
        'ISS HICO L1B hyperspectral radiance (Ligurian Sea)'),

    # PRISMA (ASI)
    # The per-pixel geometric angles live in a sibling Level2C companion,
    # which the archive does not hold for this granule; tests synthesize one.
    'PRISMA_L1': Sample(
        _CEPH / 'PRISMA' / 'PRS_L1_STD_OFFL_20210721102700_20210721102705_0001.he5',
        'PRISMA L1 STD hyperspectral (VNIR+SWIR, 30 m); requires a sibling '
        'Level2C companion for the per-pixel angles'),

    # DIMAP strips (staged under downloads/eoread)
    'GEOSAT_DE1': Sample(
        _CEPH / 'downloads' / 'eoread' / 'GEOSAT-1',
        'GEOSAT-1 (DE-1) L1 strip'),
    'GEOSAT_DE2': Sample(
        _CEPH / 'downloads' / 'eoread' / 'GEOSAT-2',
        'GEOSAT-2 (DE-2) L1C strip'),
    'PLEIADES_MS': Sample(
        _CEPH / 'downloads' / 'eoread' / 'PLEIADES' /
        'Polygon1_SO24012539-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298',
        'Pleiades multi-spectral L1 strip'),
    'PNEO_MS': Sample(
        _CEPH / 'downloads' / 'eoread' / 'PLEIADES-NEO' /
        'WO_000194880_1_1_SAL24178541-1_ACQ_PNEO4_03432708887687',
        'Pleiades-Neo multi-spectral L1 strip'),
    'PLANETSCOPE_MS': Sample(
        _CEPH / 'downloads' / 'eoread' / 'PLANETSCOPE' / 'PSScene_Basic_Analytic_8b_udm2',
        'PlanetScope Basic Analytic 8-band L1 scene'),

    # GOES-NG (CMS/ICARE "Emultic" composition)
    'GOESNG_L1': Sample(
        _CEPH / 'downloads' / 'eoread' / 'GOESNG' /
        'Emultic1kmNC4_goes16_202206171400.nc',
        'GOES-16 ABI L1b 1 km/500 m multi-channel (Emultic)'),

    # HYPSO-1 (NTNU hyperspectral)
    'HYPSO_L1': Sample(
        _CEPH / 'downloads' / 'eoread' / 'HYPSO' /
        'vancouver_2022-07-30_1825Z-l1b.nc',
        'HYPSO-1 L1B hyperspectral radiance (Vancouver)'),
}


def get(name: str) -> Sample:
    """Return the registered :class:`Sample` called ``name``."""
    if name not in _samples:
        known = ', '.join(sorted(_samples)) or '(none registered)'
        raise KeyError(f'unknown sample product {name!r} (known: {known})') from None
    return _samples[name]


def resolve(name: str) -> Path:
    """Resolve a registered sample product to an existing path."""
    return get(name).resolve()


def available() -> dict[str, Sample]:
    """Return a copy of the registry, for discovery and documentation."""
    return dict(_samples)
