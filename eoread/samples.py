#!/usr/bin/env python3
"""
Sample-product registry.

Reader classes only declare *which* sample products they support via their
``sample_products`` attribute.  This module owns the out-of-reader details:
the actual location of each sample product.

Samples are declared as explicit paths (relative to the repository for the
bundled products, absolute for the shared Ceph archive).  No environment
variable and no download is involved: a missing sample raises a loud,
actionable error.
"""

from dataclasses import dataclass
from pathlib import Path


#: Shared sample archive (developer machines).
_CEPH = Path('/mnt/ceph/samples')

@dataclass(frozen=True)
class Sample:
    """A sample product available for one (sensor, level, product type).

    Parameters
    ----------
    path : Path
        Location of the sample file or product directory.
    description : str, optional
        Human-readable description used in listings and error messages.
    """

    path: Path
    description: str = ''

    def resolve(self) -> Path:
        """Return the sample path, or raise ``FileNotFoundError``."""
        path = Path(self.path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f'{self.description or "sample"}: sample path does not exist: {path}')
        return path


#: Registry keyed by stable sample-product IDs.  Reader classes reference these
#: IDs from their ``sample_products`` attribute; they never contain paths.
_samples: dict[str, Sample] = {
    # Sentinel-2 MSI
    'MSI_L1C': Sample(
        _CEPH / 'SENTINEL-2-MSI' / 'S2B_MSIL1C_20250320T104639_N0511_R051_T31UDS_20250320T142408.SAFE',
        'Sentinel-2 MSI L1C'),
    'MSI_L2A': Sample(
        _CEPH / 'SENTINEL-2-MSI' / 'S2B_MSIL2A_20250930T102739_N0511_R108_T31TFJ_20250930T142814.SAFE',
        'Sentinel-2 MSI L2A'),

    # Sentinel-3 OLCI
    'OLCI_L1_EFR': Sample(
        _CEPH / 'SENTINEL-3-OLCI' / 'S3A_OL_1_EFR____20160720T093226_20160720T093526_20241003T153523_0179_006_307_1980_MAR_R_NT_004.SEN3',
        'Sentinel-3 OLCI L1 EFR'),
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

    # ECOSTRESS
    'ECOSTRESS_L1C': Sample(
        _CEPH / 'ECOSTRESS' / 'ECOv002_L1CG_RAD_28710_008_20230730T032333_0710_01.h5',
        'ECOSTRESS L1C radiance'),
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

    # VENµS
    'VENUS_L1C': Sample(
        _CEPH / 'VENUS' / 'VENUS-XS_20230116-112657-000_L1C_VILAINE_C_V3-1',
        'VENµS L1C'),
    'VENUS_L2A': Sample(
        _CEPH / 'VENUS' / 'VENUS-XS_20230116-112657-000_L2A_VILAINE_C_V3-1',
        'VENµS L2A'),

    # PACE OCI
    'PACE_L1B': Sample(
        _CEPH / 'PACE_OCI' / 'PACE_OCI.20250430T130447.L1B.V3.nc',
        'PACE OCI L1B'),

    # GCOM-C SGLI
    'SGLI_L1B': Sample(
        _CEPH / 'SGLI' / 'GC1SG1_201912050159F05712_1BSG_VNRDQ_1007.h5',
        'GCOM-C SGLI L1B'),

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
