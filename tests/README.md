# Test data policy

Tests run on **real products only** — there is no synthetic fixture factory.
Each test that needs data reads a path from an environment variable via
`require_sample()` (`tests/conftest.py`). If the variable is unset or the
path does not exist, the test is **skipped**, never errored: a machine without
samples must produce clean skips, and the no-data gate must stay green on
every commit.

The library never downloads data; `get_sample()` helpers in the legacy modules
are test support only and are no longer called from the test suite.

## Layout

- `tests/` — v2 conformance tests (Phase 1 of the refactor and onward).
- `tests/old/` — legacy v1 reader regression suite, kept as-is during the
  coexistence period while readers are ported one by one.
- `conftest.py` and `generic.py` are shared by both suites and stay at the root.

## Data manifest

| Env var | Sample product | Used by |
|---|---|---|
| `LEVEL1_MSI` | Sentinel-2 MSI Level-1C `.SAFE` | test_msi.py, test_autodetect.py |
| `LEVEL2_MSI` | Sentinel-2 MSI Level-2A `.SAFE` | test_msi.py, test_autodetect.py |
| `LEVEL1_OLCI` | Sentinel-3 OLCI Level-1 (SEN3) | test_olci.py |
| `LEVEL2_OLCI` | Sentinel-3 OLCI Level-2 WFR water product (SEN3) | test_olci.py |
| `LEVEL2_OLCI_WRR` | Sentinel-3 OLCI Level-2 WRR water product (SEN3) | test_olci.py |
| `LEVEL1_MODISA_MYD` | Aqua MODIS Level-1 HDF | test_modis.py |
| `LEVEL1_ECOSTRESS` | ECOSTRESS L1C HDF5 | test_ecostress.py |
| `LEVEL1B_ECOSTRESS` | ECOSTRESS v001 L1B_RAD swath radiance (TRISHNA maquette; paired L1B_GEO in same dir) | test_ecostress.py |
| `LEVEL2_ECOSTRESS` | ECOSTRESS v001 L2_LSTE (paired L1B_GEO in same dir) | test_ecostress.py |
| `LEVEL2_ECOSTRESS_V002` | ECOSTRESS ECOv002 L2_LSTE (paired ECOv002_L1B_GEO in same dir) | test_ecostress.py |
| `LEVEL1_L8`, `LEVEL2_L8` | Landsat 8 OLI L1/L2 (Collection 2) | test_landsat8.py |
| `LEVEL1_L9`, `LEVEL2_L9` | Landsat 9 OLI L1/L2 (Collection 2) | test_landsat9.py |
| `LEVEL1_MERIS` | MERIS Level-1 | test_meris.py |
| `LEVEL1_SGLI` | SGLI Level-1 | test_sgli.py |
| `LEVEL1_PACE` | PACE OCI Level-1B | test_pace.py |
| `LEVEL1_VENUS`, `LEVEL2_VENUS` | VENµS L1C / L2A | test_venus.py |
| `GOESNG_L1` | GOES-16 ABI L1b Emultic (1 km + 500 m + ICARE static grid + optional CMA cloud mask) | test_goesng.py |
| `HYPSO_L1` | HYPSO-1 L1B hyperspectral product (Vancouver sample) | test_hypso.py |
| `LEVEL1_HAWKEYE` | HAWKEYE L1A NetCDF | test_nasa.py |
| `LEVEL1_SEAWIFS` | SeaWiFS L1A NetCDF | test_nasa.py |
| `LEVEL1_MODISA` | Aqua MODIS L1A (LAC) | test_nasa.py |
| `LEVEL1_VIIRS_SNPP` | VIIRS SNPP L1A NetCDF | test_nasa.py |

## Sample locations (developer machines)

- `data/sample_products/` — MSI, PACE_OCI, OLCI FR/RR, VENUS, ECOSTRESS L1C
- `/mnt/ceph/samples/` — IASI, LANDSAT_OLI, MERIS, MODIS_AQUA, PACE_OCI, SeaWiFS, SGLI, VIIRS
- `/mnt/ceph/data/ECOSTRESS/L2_LSTE/` — ECOSTRESS ECOv002 L2_LSTE granules (paired L1B_GEO in the same dir)
- `/mnt/ceph/proj/CNES_TRISHNA_LST_MAQUETTE/2024/raw_data/india/` — ECOSTRESS v001 L1B_RAD / L1B_GEO / L2_LSTE (TRISHNA maquette)
- HYPSO: sample staged as `HYPSO_L1` (symlink to the `vancouver_2022-07-30` L1B); `test_hypso.py` skips cleanly if it is missing.

## No-data gate

```bash
pixi run -e default gate        # import smoke-check of all modules + pytest --collect-only
```

Must be green on every commit; it runs without any sample data.
