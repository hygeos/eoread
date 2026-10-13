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
| `LEVEL2_OLCI` | Sentinel-3 OLCI Level-2 (SEN3) | test_olci.py |
| `LEVEL1_MODISA_MYD` | Aqua MODIS Level-1 HDF | test_modis.py |
| `LEVEL1_ECOSTRESS` | ECOSTRESS L1B HDF5 | test_ecostress.py |
| `LEVEL1_L8`, `LEVEL2_L8` | Landsat 8 OLI L1/L2 (Collection 2) | test_landsat8.py |
| `LEVEL1_L9`, `LEVEL2_L9` | Landsat 9 OLI L1/L2 (Collection 2) | test_landsat9.py |
| `LEVEL1_MERIS` | MERIS Level-1 | test_meris.py |
| `LEVEL1_SGLI` | SGLI Level-1 | test_sgli.py |
| `LEVEL1_PACE` | PACE OCI Level-1B | test_pace.py |
| `LEVEL1_VENUS`, `LEVEL2_VENUS` | VENµS L1C / L2A | test_venus.py |
| `DIR_SAMPLE_HYPSO` | HYPSO sample directory (no local copy known) | test_hypso.py |
| `LEVEL1_HAWKEYE` | HAWKEYE L1A NetCDF | test_nasa.py |
| `LEVEL1_SEAWIFS` | SeaWiFS L1A NetCDF | test_nasa.py |
| `LEVEL1_MODISA` | Aqua MODIS L1A (LAC) | test_nasa.py |
| `LEVEL1_VIIRS_SNPP` | VIIRS SNPP L1A NetCDF | test_nasa.py |

## Sample locations (developer machines)

- `data/sample_products/` — MSI, PACE_OCI, OLCI FR/RR, VENUS
- `/mnt/ceph/samples/` — ECOSTRESS, IASI, LANDSAT_OLI, MERIS, MODIS_AQUA, PACE_OCI, SeaWiFS, SGLI, VIIRS
- HYPSO: no copy anywhere yet → `test_hypso.py` always skips until one is provided.

## No-data gate

```bash
pixi run -e default gate        # import smoke-check of all modules + pytest --collect-only
pixi run -e xarray-2024 gate    # same, against the older supported xarray generation
```

Both must be green on every commit; they run without any sample data.
