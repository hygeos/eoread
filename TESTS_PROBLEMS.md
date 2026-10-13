# Test suite problem catalog

Catalog of every problem the test suite points at, from a full baseline run of
`pixi run pytest tests/` (227 tests collected): **70 failed, 93 passed, 16 skipped, 48 errors**.

Full raw log: `/tmp/opencode/pytest_full.txt` (`--tb=line`).

Failures group into **9 root-cause clusters** (A–I). Each test below is accounted for
(23+4+2+11+20+13+13+16+16 = 118 failing tests).

## Re-baseline after Phase 0 (legacy migration + no-data gate)

Full run of `pixi run pytest tests/` under identical conditions (same samples,
same env vars): **31 failed, 89 passed, 102 skipped, 5 errors**
(raw log `/tmp/opencode/pytest_final.txt`).

- The remaining failures are a **strict subset** of the baseline set — no new
  failures. 82 of the 118 baseline failures now pass or skip cleanly.
- Errors 48 → 5: missing data/env now produces clean skips via
  `require_sample()` in `tests/conftest.py`. The only regressions caused by
  the move (6 `flag_reader` tests, broken `_flag_reader` import strings) were
  fixed with a legacy import fallback in `eoread/flags.py`.
- Remaining 36 failing tests: goesng 16 (cluster I, deferred WIP), modis 11
  (cluster D), landsat8/9 `test_l1c_main` 4 (cluster B), nasa SEAWIFS 4
  (cluster H), sgli `test_time` 1 (cluster E flake). Cluster A (MSI drop_vars)
  passes via the legacy stopgap; clusters F/G (PACE, HYPSO missing data) and
  most of H are now clean skips.
- No-data gate (`pixi run gate` = import smoke-check + `pytest --collect-only`)
  is green in both the `default` and `xarray-2024` pixi environments.

| # | Cluster | Category | Tests |
|---|---------|----------|-------|
| A | MSI geometry interpolation `drop_vars` bug | **real reader bug** | 21 msi + 2 autodetect = 23 |
| B | Landsat-8/9 `test_l1c_main` generic check failure | suspected real bug (msg truncated) | 4 |
| C | ECOSTRESS latitude ordering | suspected real bug (msg truncated) | 2 |
| D | MODIS netCDF4/HDF4 ImportError, no `load_hdf4` fallback | reader + env | 11 |
| E | Intermittent HDF5 runtime I/O errors (flaky, not reproducible alone) | environment / flaky | 20 |
| F | PACE sample file missing on `/mnt/ceph` | data / env | 13 |
| G | HYPSO: no sample + undefined env var (+ latent test-API misuse) | data / env + test code | 13 |
| H | NASA: env-var name mismatches, OCSSW/docker absent | environment | 16 |
| I | GOESNG known-broken WIP (explicitly deferred) | deferred by decision | 16 |

---

## A. MSI — `drop_vars(['y','x'])` fails after tie-point interpolation (23 tests)

**Error:** `ValueError: These variables cannot be found in this dataset: ['y', 'x']`
at `eoread/msi.py:680`, reached from `Level1_MSI.__init__` → `_Internal.read_geometry`
(`msi.py:147`).

Both L1C and L2A paths fail, for every `chunks` × `resolution` combination:

- FAILED: `test_instantiation[10-500]`, `[10-chunks1]`, `[None-500]`, `[None-chunks1]`,
  `test_l2_instantiation[500]`, `[chunks1]`, `test_main[500]`, `[chunks1]`, `test_plot`,
  `test_subset[500]`, `[chunks1]`, `test_time[500]`, `[chunks1]`
- ERROR (fixture setup): `test_flag_reader[500-10]`, `[500-None]`, `[chunks1-10]`, `[chunks1-None]`,
  `test_lazy_load[500-10]`, `[500-None]`, `[chunks1-10]`, `[chunks1-None]`

**Mechanism (from traceback):** in `_Internal.read_geometry` (`msi.py:661-680`), solar/view
angles are interpolated from tie points with `interp(..., tie_rows=Linear(y), tie_columns=Linear(x))`
where the target `DataArray`s carry dims named `y`/`x`. The interpolated result has dims `y`/`x`
but **no coordinate variables** by those names, so the subsequent `.drop_vars([d['y'], d['x']])`
raises. Looks like a behavior change in `xarray.interp` between the version the code was written
for and the one installed (newer xarray does not copy the target's index as a coord).

**Also fails through this path:** `tests/test_autodetect.py::test_level1`, `test_level2`
(`autodetect` dispatches to `Level1_MSI`/`Level2_MSI` on the MSI samples) — same
`ValueError: These variables cannot...`.

This is the single biggest cluster and is the most likely "real bug" of the whole suite.

---

## B. Landsat-8 / Landsat-9 — `test_l1c_main` generic check (4 tests, suspected real bug)

**Error (truncated in log):** `ValueError: Dataset shoul...` from
`generic.Test.main` (`tests/generic.py`) — one of its dataset-shape/variable checks fails.

- FAILED: `tests/test_landsat8.py::test_l1c_main[500]`, `[chunks1]`
- FAILED: `tests/test_landsat9.py::test_l1c_main[500]`, `[chunks1]`

Notable: the *other* Landsat tests (instantiation, time, subset, lazy_load, flag_reader, plot)
**pass**, so the OLI reader works — only the full `Test.main` invariants fail. Same failure on
both L8 and L9 samples → systematic issue in the OLI reader output (or in what `Test.main`
asserts). Full message needs one look at the log to confirm which check trips.

---

## C. ECOSTRESS — latitude ordering (2 tests, suspected real bug)

**Error (truncated):** `ValueError: Latitude is or...` (ordering-related check).

- FAILED: `tests/test_ecostress.py::test_latlon[500]`, `[chunks1]`

All other ecostress tests pass → specific to the lat/lon generation in `eoread/ecostress.py`.
(Consistent with the known `_v1_compat` mess in that module; the lat/lon path is what
`test_latlon` exercises.)

---

## D. MODIS — ImportError: netCDF4 built without HDF4, no fallback (11 tests)

**Error:** `ImportError: This error occurs because ...` — the pixi env's `netcdf4` was compiled
without HDF4 support; `eoread/modis.py` raises this ImportError instead of falling back to the
`load_hdf4` helper that already exists in `eoread/hdf4.py` for exactly this case.

- FAILED: `test_l1c_instantiation[200-500]`, `[200-None]`, `[chunks1-500]`, `[chunks1-None]`,
  `test_l1c_time[200]`, `[chunks1]`, `test_plot`
- ERROR: `test_flag_reader`, `test_l1c_lazy_load`, `test_l1c_main`, `test_l1c_subset`

Sample exists (`LEVEL1_MODISA_MYD=/mnt/ceph/samples/MODIS_AQUA/MYD021KM....hdf`).
Fix candidates: (a) make `modis.py` fall back to `eoread.hdf4.load_hdf4`, or (b) rebuild the
netcdf4 dep with HDF4. (a) is the robust one — any consumer without HDF4 support hits this.

---

## E. Intermittent HDF5 runtime I/O errors — flaky, not reproducible in isolation (20 tests)

**Errors:** `RuntimeError: Unspecified error in H5D...` (H5Dread) and
`OSError: [Errno -101] NetCDF: H...` (errno −101 = `NC_EHDFERR`).

Affected:
- `tests/test_sgli.py`: ERROR `test_flag_reader[500/chunks1]`, `test_lazy_load[500/chunks1]`,
  `test_main[chunks1]`; FAILED `test_plot`, `test_subset[500/chunks1]`, `test_time[500/chunks1]` (10)
- `tests/test_olci.py`: ERROR `test_flag_reader[500/chunks1]`, `test_l1c_lazy_load[500/chunks1]`,
  `test_l1c_subset[chunks1]`; FAILED `test_l1c_subset[500]`, `test_level2[500/chunks1]` (8)
- `tests/test_venus.py::test_l1c_subset[500]` (1)
- `tests/test_tools.py::test_open_raster` (1)

**Evidence this is an environment/load issue, not a reader bug:**
- The exact SGLI code path (`xr.open_datatree(p, engine='h5netcdf', phony_dims='sort')` →
  `tree['Image_data'].to_dataset()` → full read of `Lt_VN01`) **runs clean in isolation**.
- `test_open_raster`'s exact sequence (write `.nc` with h5netcdf, open via `open_raster`/GDAL)
  **passes when run alone**.
- Plain h5netcdf write+read of a small file works fine.
- Cluster membership **varies between two consecutive full runs** (67F/47E vs 70F/48E; e.g.
  olci `test_l1c_time`/`test_l1c_instantiation[chunks1]` failed in run 1 but not run 2, while
  `test_l1c_subset[500]`/venus joined in run 2) → intermittent.

Likely cause: concurrent HDF5 I/O under `pytest-xdist -n auto` (in `pytest.ini` addopts), several
workers hammering large files on the `/mnt/ceph` network filesystem at once. Not a code defect;
candidate mitigations are env-level (HDF5 VFD settings, fewer workers for these tests) — out of
scope for the refactor steps unless it keeps masking real results.

---

## F. PACE — sample file missing (13 tests)

**Error:** `FileNotFoundError: [Errno 2] No such file or directory` on open.

- FAILED: `test_instantiation[500/chunks1]`, `test_main[500/chunks1]`, `test_plot`,
  `test_subset[500/chunks1]`, `test_time[500/chunks1]`
- ERROR: `test_flag_reader[500/chunks1]`, `test_lazy_load[500/chunks1]`

`LEVEL1_PACE=/mnt/ceph/samples/PACE_OCI/PACE_OCI.20250101T000738.L1B.V3.nc` is set, so
`get_sample()` returns that path directly and the reader fails to open it — the file (or its
container) is not actually present/readable on ceph. Data problem, not a test or code problem.
(Local copy exists at `data/sample_products/PACE_OCI`, 1.6 G — env var could point there.)

---

## G. HYPSO — no sample + undefined env var; latent test-API misuse (13 tests)

**Error:** `KeyError: 'DIR_SAMPLE_HYPSO is not defined, and no default has been provided'`
in the `level1_hypso` fixture (`get_sample()` in `eoread/hypso.py`). All 13 tests error at
fixture setup — there is no HYPSO sample anywhere (not in `DIR_SAMPLES`, not on ceph), and the
env var the lookup needs is not defined.

Tests: `test_instantiation[500/chunks1]`, `test_main[500/chunks1]`, `test_plot[500/chunks1]`,
`test_subset[500/chunks1]`, `test_time[500/chunks1]`, `test_lazy_load[500/chunks1]` (ERROR),
`test_v1_compat`.

**Latent test-code bugs hidden behind the missing data** (will explode as soon as a sample
exists — see static audit below): the module calls nonexistent module-level functions
`generic.test_main`, `generic.test_execution_time`, `generic.test_lazy_load`, `generic.test_read`,
`generic.test_subset`, `generic.test_plot`. The real API is `generic.Test.main(...)`,
`generic.Test.execution_time(...)`, etc., plus module-level `generic.plot(...)`.

Also: `test_v1_compat` reads `getdir("DIR_V1_COMPAT_DATA")` — an env var that is **not defined**
in this environment (verified: lookup raises). Even with sample data, that test cannot run.

---

## H. NASA (`makeL1C`) — env-var mismatches + OCSSW/docker absent (16 tests)

All failures in `tests/test_nasa.py` are environmental:

1. **Env-var name mismatch (8 ERROR):** the fixture dict asks for `LEVEL1_MODISA` and
   `LEVEL1_VIIRS_SNPP`, but this shell defines `LEVEL1_MODISA_MYD` and `LEVEL1_VIIRS` →
   `KeyError: 'LEVEL1_MODISA is not defined'`, `'LEVEL1_VIIRS_SNPP is not defined'`.
   Affected: `test_instantiate[MODIS/VIIRS]`, `test_L1C[MODIS|VIIRS-docker|shell]`,
   `test_plot[MODIS/VIIRS]`.
2. **Empty HAWKEYE var (4 FAILED):** `LEVEL1_HAWKEYE` is set to an *empty string*, so the
   fixture's `assert Path(getvar(...)).exists()` passes (`Path('') → '.'`), then
   `makeL1C('.')` raises `ValueError: Invalid sensor in makeL1C`.
   Affected: `test_instantiate[HAWKEYE]`, `test_L1C[HAWKEYE-docker|shell]`, `test_plot[HAWKEYE]`.
3. **OCSSW not installed (3 FAILED):** `KeyError: 'OCSSWROOT is not defined'` for the `shell`
   method, `ModuleNotFoundError` for the `docker` method. Affected:
   `test_instantiate[SEAWIFS]`, `test_L1C[SEAWIFS-shell]`, `test_plot[SEAWIFS]`,
   `test_L1C[SEAWIFS-docker]`.

The SEAWIFS sample itself exists (`/mnt/ceph/samples/SeaWiFS/S2004115125135.L1A_GAC.Z`) — only
the L1A→L1C conversion toolchain is missing. Tests need `pytest.skip` guards for
missing env vars / OCSSW instead of hard errors (test-code fix, step 5).

---

## I. GOESNG — known-broken WIP module (16 tests) — *deferred by explicit decision*

`eoread/goesng.py` + `tests/test_goesng.py` are left broken on purpose; no fixes planned.
Documented here for completeness:

- **NameError: name 'local_config' is not defined** (13): `test_clouds`,
  `test_instantiate[1000]`, `[chunksize1]`, `test_main`, `test_subset`, and all 8
  `test_preview[...]` params. A module-level name the tests reference that was never defined
  (or renamed).
- **pyhdf.error.HDF4Error: SD: no s...** (1): `test_load_hdf4` — HDF4 open error on the sample.
- **ERROR `test_read`** (1): message truncated in log (`file /home/nathan/proj/eoread/tests/t...`)
  — a file-related error, not investigated further per the deferral decision.

---

# Latent test-code defects (not failing today, will be)

These don't show up as failures because their tests are skipped or masked by cluster G/H,
but they were found in the static audit and must be fixed when the data/env issues clear:

1. **`tests/test_olci.py` — duplicate `test_plot`.** Defined at :20 (band `'Oa09'`) and again
   at :27 (`v1_compat=True`, band `865`). The second definition *shadows* the first, so only the
   v1_compat variant ever runs. They test different things → rename one
   (e.g. `test_plot_v1_compat`), don't delete.
2. **`tests/test_hypso.py` — wrong generic API.** Six calls to nonexistent module-level
   `generic.test_*` functions; real API is `generic.Test.*` and module-level `generic.plot`
   (see cluster G).
3. **Dead skipped tests (3).** `test_read` in `test_msi.py:49-56`, `test_hypso.py:45-51`,
   `test_sgli.py:43-48`: all `@pytest.mark.skip()`, reference a nonexistent `param` fixture and a
   nonexistent `generic.test_read` / `Test.read` helper; the MSI one also uses unimported `eo`.
   Nothing generic implements per-index reads → recommend deleting all three. (16 tests are
   skipped in total.)
4. **Fixture scope inconsistency.** Sample-path fixtures are session-scoped in
   msi/ecostress/landsat8/landsat9/venus, module-scoped in hypso/meris/sgli/modis/olci/autodetect,
   and *function*-scoped in pace (`level1_oci` re-runs `get_sample()` per test). Normalize to
   session.
5. **`tests/test_pace.py::test_main`** calls `.compute()` on the 1.6 G product before passing it
   to `generic.Test.main`, whose assertions don't compute — pure slowdown, and it defeats the
   point of the chunk-consistency check. Function-scoped fixtures compound this (every test
   re-opens the dataset).

# Environment inventory (context for clusters D–I)

| Resource | State |
|----------|-------|
| `DIR_SAMPLES` | `data/sample_products/` — has ECO, PACE_OCI (1.6 G), SENTINEL-2-MSI (2.5 G), OLCI-FR (983 M), OLCI-RR (449 M), VENUS (9.8 G); **no HYPSO** |
| `/mnt/ceph/samples` | ECOSTRESS, IASI, LANDSAT_OLI, MERIS, MODIS_AQUA, PACE_OCI, SeaWiFS, SENTINEL-2-MSI, SENTINEL-3-OLCI, SGLI, VIIRS — readable (HDF5 flakiness under load, cluster E) |
| `LEVEL1_PACE` | points at ceph path that is **not openable** (cluster F) |
| `LEVEL1_HAWKEYE` | set to empty string (cluster H.2) |
| `LEVEL1_MODISA_MYD`, `LEVEL1_VIIRS` | set, but tests ask for `LEVEL1_MODISA` / `LEVEL1_VIIRS_SNPP` (cluster H.1) |
| `DIR_V1_COMPAT_DATA` | **undefined** — `test_hypso.py::test_v1_compat` can never run |
| OCSSW (`OCSSWROOT`), docker module | absent (cluster H.3) |
| netCDF4 in pixi env | built **without HDF4** support (cluster D) |
| `pytest.ini` | `addopts = -n auto --html=...` — xdist parallelism implicated in cluster E flakiness |

# Suggested mapping to the refactor plan (`TODO.md`)

- Clusters **A, B, C, D** → real bugs to fix under step 2 (A is new and is the biggest one;
  D = `modis.py` should fall back to `eoread/hdf4.load_hdf4`).
- Cluster **E** → environment; leave alone unless it keeps masking A–D verification.
- Clusters **F, G(data), H** → data/env: point env vars at existing samples, add skip guards
  (test side, step 5).
- Cluster **I** → stays deferred per decision.
- Latent list → all step 5 (tests rewrite).
