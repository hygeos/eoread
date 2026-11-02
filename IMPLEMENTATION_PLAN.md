# eoread v2 — implementation plan

Companion to `REFACTORING.md` (agreed design). This plan incorporates every decision from the 2026-10-13 design sessions. Phases are sequential; each phase ends in a commit and a green no-data gate.

## Decisions incorporated (reference)

- Lean plain Python: `Reader` base + shared transforms + `schema.validate()` — no spec engine.
- Level = class, type = variant table (metadata-detected), version = explicit if/elif + loud raise.
- Naming: level explicit — `Level1C_MSI`, `Level2A_MSI`, `Level1C_Ecostress`.
- Day one: all current reader modules move to `eoread/legacy/`; **no re-exports** (deliberate import break).
- Stopgaps on legacy code: minimal safe user-visible fixes only.
- v1-compat: one shared `eoread/compat.py::v1(ds)` over the canonical schema; pure rename/reorg = compat, value differences = documented clean break per reader.
- Phase 1 sensors and order: **MSI → ECOSTRESS → OLCI → MODIS → OLI**. Phase 2 (later): VENUS (**ported + parity-tested — see prep notes below**), PACE (**ported + parity-tested — see prep notes below**), MERIS (**ported + parity-tested — see prep notes below**), SGLI (**ported + parity-tested — see prep notes below**), the four WIP DIMAP readers geosat/planetscope/pleiades/pleiades_neo (**ported + parity-tested — see prep notes below**), GOESNG and HYPSO (**ported on user request — see prep notes below**); NASA L1A stays open. The "DIMAP deadline via VENUS" premise is **retired**: VENUS V3-1 is MTD XML + GeoTIFF, not DIMAP (see the REFACTORING.md decision log).
- Chunk defaults: values unchanged per reader, declared as `chunks_default` class attr.

## Canonical schema (locked)

- **Registry**: `core.geo.naming.names` is the single source of truth; `eoread/schema.py::validate(ds)` checks presence/dims/dtype/ranges from it, and a payload `unit` is only reported when the product metadata states one. New names are proposed upstream in `core` before a port needs them.
- **Dims**: `y`, `x` (pixel indices), `bands` (sensor-native string labels: `B01…`, `b01…`). One grid per dataset.
- **Multi-resolution**: single grid; `resolution` param chooses target (default = finest); native per-res output is an opt-in mode. Coarse bands upsample lazily via dask — no data lost.
- **Payload**: exactly one *primary* payload var from `{Ltoa, Rtoa, BT}`, declared per class; extra declared payload vars allowed where the product natively carries them (MODIS: `Rtoa` alongside primary `Ltoa`). Radiance products emit `Ltoa` (fixes MSI L1C misnomer); reflectance keeps `Rtoa`.
- **Geolocation**: 2D vars `latitude`, `longitude` (degrees, float32) over `(y,x)` — mandatory.
- **Angles**: `sza`, `saa`, `vza`, `vaa` — 2D full grid, degrees, float32. Raw tie-point data allowed as optional `*_tie` extras.
- **Bands**: string labels + `cwav` var (nm); optional `bands_group` coord.
- **Attrs required** (validate fails without them): `sensor`, `platform`, `product_name`, `datetime`, `input_directory`, `resolution`, `shortname`, `crs`, plus new `schema_version` (stamped by `finalize()`). Where a product lacks e.g. a CRS string, the reader derives it (e.g. from UTM zone).
- **Flags**: `_flag_reader` attr only; decode on demand. Readers supporting flags declare `flags = True` and implement `flag_requires` / `flag_dims_like` / `getflag` as classmethods; no flags/quality vars in the dataset.
- **Tests**: real samples only (no synthetic fixtures); clean skip when data/env missing; one documented data manifest.

---

## Phase 0 — skeleton + legacy move

1. **Legacy move.** `git mv` all sensor reader modules into `eoread/legacy/`; strip re-exports from `__init__.py`. `tools.py` shared helpers that the new code will use are extracted (below); legacy-only helpers move with the readers. `flags.py` stays at top level (new and legacy both import it).
2. **Stopgap.** Apply the one-line MSI `drop_vars(['y','x'])` fix in `legacy/msi.py` (cluster A, 100% broken) — the only legacy bug fix permitted in this phase.
3. **`eoread/reader.py`.** `Reader` base: fixed signature `(path, chunks=None, metadata_template=None, verbose=False)`; class attrs `sensor`, `level`, `chunks_default`, `required_vars`, `payload_var(s)`; `_finalize(ds, metadata)` = attach filtered metadata + `_flag_reader` (the reader class path, when `flags = True`) + `schema_version` → `schema.validate(ds)`. Exceptions `UnsupportedProductVersion`, `UnsupportedProductType`. Generic variant support: optional `VARIANTS` table + `detect_variant()` (metadata-based, loud raise on unknown).
4. **`eoread/schema.py`.** `validate(ds)` per the locked schema above.
5. **`eoread/transforms/` package.** One implementation per operation, extracted from `tools.py` + per-reader copies: `latlon_utm`, `tiepoint_angles`, single-grid resample to target resolution (from MSI's `spatial_resample` usage), radiometry helpers.
6. **`eoread/compat.py`.** `v1(ds)` shared transform scaffold (rename/reorg over the canonical schema); per-reader clean-break doc section.
7. **Test harness.** Rework `tests/generic.py`: geometry invariants (lat/lon monotonicity & ordering, angle ranges), laziness invariant (dask graph intact after instantiation), `validate()` conformance. Skip guards in `tests/conftest.py` driven by a data manifest; write `tests/README.md` (env vars + sample locations). No-data gate: import smoke of all modules + `pytest --collect-only`.
8. **CI matrix** over supported xarray/dask/netcdf versions (pixi task).

**Done when:** `eoread/readers/` exists empty; no-data gate green; legacy suite re-baselined — the 48 error cascades are now clean skips, pass/fail otherwise unchanged from the 2026-10-13 baseline.

## Phase 1 — MSI (`Level1C_MSI`, `Level2A_MSI` over `_MSIBase`)

1. `_MSIBase(Reader)`: SAFE/granule discovery, XML parsing, base attrs (platform as string), band table, `latlon_utm`, `tiepoint_angles`; single level-specific hook (`payload`).
2. `Level1C_MSI`: payload **`Ltoa`** (semantics fix); version if/elif on `PROCESSING_BASELINE` with exhaustive `else: raise UnsupportedProductVersion`.
3. `Level2A_MSI`: payload `Rtoa`; drop B8/B10; single-grid default + `resolution` param + native per-res opt-in mode.
4. `compat.v1` mapping for MSI (old names/shapes → v1) or documented clean break, per the rule.
5. Tests: `tests/test_msi.py` on the local 2.5 G samples through the generic harness; retire old MSI tests after parity check.

**Done when:** both classes pass the harness (geometry, laziness, validate) on real L1C and L2A samples; compat verified or break documented; `legacy/msi.py` + old tests deleted.

## Phase 2 — ECOSTRESS (`Level1C_Ecostress`)

1. Inspect the available L1C sample types; confirm the shared contract (user-confirmed: all 5 share one output); record each variant's layout diffs.
2. `_EcostressBase(Reader)`: H5 open, orbit/attitude metadata, band table.
3. `Level1C_Ecostress`: 5-entry `VARIANTS` table with metadata-field detection; loud raise on unknown type. (Port any other ECOSTRESS level currently in active use — check during the phase.)
4. Compat mapping or documented break.

**Done when:** every available L1C variant reads through the one class and passes the harness; an unknown/foreign product raises at open time; `legacy/ecostress.py` + old tests deleted.

## Phase 3 — OLCI (L1 + L2)

**Status: DONE** — ported and harness-green on the local OLCI-FR (EFR + WFR) and OLCI-RR (WRR) samples; `legacy/olci.py` retained as the reference implementation and the old suite kept at `tests/old/test_olci.py`, consistent with Phases 1–2.

1. Ported both levels over a shared `_OlciBase(Reader)` (`eoread/readers/olci.py`): variant detection from the manifest `generalProductInformation.productType` (`OL_1_EFR___` / `OL_2_WFR___` / `OL_2_WRR___`) plus a level check on the manifest text; loud `UnsupportedProductType` at open for unknown types; nominal resolution per variant (EFR/WFR 300 m, WRR 600 m).
2. L1 → payload `Ltoa` (units data-driven from the band files) + lazy declared extra `Rtoa`; quality flags via `Level1_OLCI.getflag`. L2 → primary **`rho_w`** (user-confirmed), `uncertainty` as declared extra; no flag reader (WQSF bits not in the core registry).
3. The legacy `cwvl` length mismatch is dead by construction: the band table comes from the manifest `bandDescriptions` order and is cross-checked loudly against the files on disk at open time.
4. Compat: `compat.v1` is an identity for OLCI (strips `schema_version` only).

**Done when:** harness passes on local OLCI-FR (983 M) and OLCI-RR (449 M) samples; `legacy/olci.py` + old tests deleted.

## Phase 4 — MODIS (`Level1B_MODIS`)

**Status: DONE** — ported and harness-green on the MODIS_AQUA ceph sample (Aqua MYD021KM), including under the loud pyhdf fallback path; documented clean break in `eoread.compat` (radiance formula fix). `legacy/modis.py` retained as the reference implementation and the old suite kept at `tests/old/test_modis.py`, consistent with Phases 1–3.

1. Engine contract with loud fallback: netCDF4 → `load_hdf4` (fixes the cluster D ImportError class).
2. Port: 36 bands, string labels, `cwav`; primary payload `Ltoa` + declared extra `Rtoa`; keep `*_tie` raw tie-point extras; single grid at target resolution (default finest = 250 m) replacing the red-channel split dims.
3. Compat mapping or documented break.

**Done when:** harness passes on the MODIS_AQUA ceph sample, including under the `load_hdf4` fallback path; `legacy/modis.py` + old tests deleted.

## Phase 5 — Landsat OLI (L8/L9)

**Status: DONE** — ported and harness-green on the LANDSAT_OLI ceph samples (L8 LC08 + L9 LC09, at both 15 m and 30 m resolutions); documented clean break in `eoread.compat` (Rtoa double solar correction removed, B8 onto the single bands grid, unsigned azimuths). `legacy/oli.py` retained as the reference implementation and the old suite kept at `tests/old/test_landsat8.py` / `test_landsat9.py`, consistent with Phases 1–4.

1. Port the level(s) currently in use (check `legacy/oli.py` scope during the phase); payload `Rtoa` for reflectance products; fix the latent `cwvl` alignment issue as part of the port, not a patch.
2. Compat mapping or documented break.

**Done when:** harness passes on the LANDSAT_OLI ceph sample; root cause of today's failing `test_l1c_main` is gone by construction; `legacy/oli.py` + old tests deleted.

## Phase 6 — consolidation

**Status: DONE** — compat final pass complete (MODIS added to `CLEAN_BREAKS` with a pinning test; per-sensor status documented in `eoread/compat.py` + REFACTORING.md §9); `pyproject.toml` synced with the new layout (deps, `requires-python`, `samples` extra); `REFACTORING.md` reflects the final state (Phase 4–6 decision-log entries); phase-2 prep notes drafted below, agreement deferred to a future session per the done-when. Packaging verified: the built wheel contains the new layout only (no tests/scripts) and byte-compiles clean; the fixed smoke gate imports all 33 modules in both matrix environments.

1. Final pass over `compat.v1` (all five sensors) + per-reader clean-break documentation.
2. Packaging: sync `pyproject.toml` with the new layout (absorbs old TODO step 6).
3. Keep `REFACTORING.md` in sync with any decisions made during implementation (it already reflects the full 2026-10-13 design state).
4. Phase-2 prep notes: VENUS/DIMAP deadline, PACE, MERIS, SGLI, GOESNG, NASA L1A scoping. GOESNG was subsequently ported on user request (see the prep note below).

**Done when:** package installs and imports clean from the new layout only; `REFACTORING.md` reflects the final state; phase-2 list agreed for a future session.

---

## Phase 2 prep notes — remaining sensors (draft for a future session)

Scoping per Phase-6 item 4. **Draft, not yet agreed** — each entry lists what exists today and the open questions to settle when the phase starts. GOESNG was ported on explicit user request (see its DONE entry); the legacy reader + `tests/old/test_goesng.py` stay broken.

### VENUS — DONE (ported + parity-tested)
- Ported: `eoread/readers/venus.py` (`Level1C_VENUS`, `Level2A_VENUS`) for V3-1 granules — MTD XML + per-band GeoTIFF rasters. **Not DIMAP**: the draft note's format assumption was wrong (the legacy reader's lazy `sand` import lives only in `get_sample`). Tests: `tests/test_venus.py` (real ceph samples, both levels; parity vs legacy + angle/latlon ground truth + exit probes — 35 passed). Legacy reader + old tests retained.
- Deviations from legacy are documented, not clean breaks (`eoread/compat.py` docstring + REFACTORING.md §9): south-up flip; new canonical tie-point angles (legacy computed none); SOL/VIE kept raw on their native tie grids — legacy's VIE reindex bug (row 23 / col 26 all-NaN) and its dropped detector bands 5–8 are fixed; L2A nodata → NaN, ATB calibrated with the MTD quantification factors, PIX_XS assigned positionally (legacy all-NaN); lat/lon from a per-block UTM transform instead of legacy's flipped corner interpolation.
- Consequence: VENUS needed no shared DIMAP machinery, so the four WIP DIMAP readers below no longer ride on a VENUS-driven deadline; their fate is decided on its own merits.

### The four WIP DIMAP readers — DONE (ported + parity-tested)
- Ported: `eoread/readers/pleiades.py` (`Level1_PLEIADES` + the shared `_DimapBase`), `pleiades_neo.py` (`Level1_PNEO`), `geosat.py` (`Level1_GEOSAT` — one class serving DE-1 and DE-2), `planetscope.py` (`Level1_PLANETSCOPE`). Tests: `tests/test_pleiades.py` (16 passed), `tests/test_pleiades_neo.py` (16), `tests/test_geosat.py` (31 + 2 variant-specific skips), `tests/test_planetscope.py` (16) — real ceph samples, parity vs legacy at pinned windows, per-product geometry ground truth (DIMAP vertex bilinear / GCP spline / UTM affine / RPC) + exit probes. Legacy readers + old tests retained.
- Importability resolved: `xmltodict` added to the `[project] dependencies` legacy group; `jrc_rayleigh_processor` is a guarded no-op stub installed by `eoread/legacy/__init__.py` (raises `NotImplementedError` if called — it only feeds SRF computation, never the read path); the Phase-6 `EXCLUDED` set is removed from `scripts/import_smoke.py`, so all 46 modules are smoked.
- The shared DIMAP glue is deduplicated in `_DimapBase` (old TODO step 7): strip discovery, nodata flags, lazy JP2/TIF opens, blockwise interpolation helpers.
- Deviations from legacy are documented, not clean breaks (`eoread/compat.py` docstring + REFACTORING.md §9): south-up flip; nodata DN==0 → NaN (legacy kept 0 as data — for GEOSAT-2 the calibration bias survives in legacy); PNEO keeps all six bands (legacy silently dropped red-edge + deep-blue); DE-2 `datetime` is the full START_TIME (legacy was date-only); `cwav` float64 (GEOSAT legacy int32, PlanetScope float32). PNEO azimuths keep the product's unsigned [0, 360) convention — documented registry inconsistency (OLCI/MSI/MERIS/PACE/SGLI precedent), checked locally in `test_pleiades_neo.py`.
- The `_srf_getter`/`_srf_getter_arg` attrs are carried over from legacy as plain strings; PNEO inherits the Pleiades getter (arg 'PNEO4'). Never invoked by the tests (network + table download).

### PACE — DONE (ported + parity-tested)
- Ported: `eoread/readers/pace.py` (`Level1B_PACE_OCI`) for OCI L1B NetCDF-4 products (the only level with a legacy reader and a sample). Tests: `tests/test_pace.py` (real ceph sample; parity vs legacy + scene-aware latlon/angle checks — 17 tests). Legacy reader + old tests retained.
- Deviations from legacy are documented, not clean breaks (`eoread/compat.py` docstring + REFACTORING.md §9): per-band chunking where legacy read one all-bands block; boolean flag masks where legacy returned raw values — `LAND` repairs a latent legacy `TypeError` on the float32 watermask; `shortname`/`crs` attrs added; `resolution` keeps legacy's hardcoded 1200 m (no data-driven spacing in the product); azimuths keep the product's signed relative [-180, 180] convention (documented registry inconsistency, OLCI/MSI/MERIS precedent). No orientation change: the product is natively south-up and legacy kept it that way.
- Environment note (also in the REFACTORING.md decision log): the reader, its test and the legacy reader all open with an explicit `h5netcdf` engine — xarray's auto-engine would pick netCDF4, whose bundled libhdf5 loaded before h5py's breaks the suite's h5netcdf write path (same pin as OLCI/ECOSTRESS).

### MERIS — DONE (ported + parity-tested)
- Ported: `eoread/readers/meris.py` (`Level1_MERIS`) for N1 products via the `epr` package (pyepr, already a declared dependency). Tests: `tests/test_meris.py` (real ceph RR sample; parity vs legacy + F0/wav ground truth + custom geolocation checks + exit probe — 19 tests). Legacy reader + old tests retained.
- Deviations from legacy are documented, not clean breaks (`eoread/compat.py` docstring + REFACTORING.md §9): south-up flip; `F0`/`wav` always emitted (legacy gated them behind `read_auxdata=False`) — a superset of the default legacy output; `resolution` from SPH `COLUMN_SPACING` instead of legacy's hardcoded 300 m (RR sample: 1040 m); `shortname`/`crs` attrs added; azimuths keep the product's absolute [0, 360) convention (documented registry inconsistency, OLCI/MSI precedent).
- The usage-vs-archival question is moot for the read path: epr stays the dependency either way.

### SGLI — DONE (ported + parity-tested)
- Ported: `eoread/readers/sgli.py` (`Level1B_SGLI`) for GCOM-C SGLI L1B HDF5 products. Tests: `tests/test_sgli.py` (real ceph sample; bit-parity vs legacy where legacy was correct + value pins against the product's raw tie data / counts — 19 tests). Legacy reader + old tests retained.
- Deviations from legacy are documented, not clean breaks (`eoread/compat.py` docstring + REFACTORING.md §9): south-up output via a **data-driven** flip (the file stores lines in acquisition order and GCOM-C tracks alternate per half-orbit; the sample granule is north-up and gets flipped); full affine `Slope*x + Offset` on the tie fields where legacy applied `(1+Slope)*x` — lat/lon were doubled and every centidegree angle inflated ~101× (confirmed bug fix, pinned against the product's raw data); float32 geometry per the naming registry; `water` is a land percentage 0–100 and legacy's `~water` on the uint8 was True on every pixel — v2 flags `LAND = water >= 50`; `L1_INVALID` = QA_flag bit 0 *not* set where legacy used the raw value (the file's own bit description says 1 = "Integrity", so the legacy test flagged the sound ~86% of the sample scene); `add_ancillary_data` dropped (instrument telemetry with no consumer; `auxdata/sgli` is now unused); `shortname`/`crs` attrs added; `datetime` normalized to ISO from the product's non-ISO `Scene_center_time` (same instant).

### GOESNG — DONE (ported on user request)
- Ported: `eoread/readers/goesng.py` (`Level1_GOESNG`) for the CMS/ICARE `Emultic` composition of GOES-16 ABI direct readout: 1 km `VIS_004`/`VIS_008`/`VIS_016`, the 500 m `VIS_006` box-averaged onto the 1 km grid, the ICARE static grid (lat/lon, viewing angles, pixel area, NetCDF or HDF4) and the optional NWC/GEO cloud mask. Solar angles are lazy pysolar at the acquisition midpoint. Tests: `tests/test_goesng.py` (40 real-data tests, two chunkings, value parity against the raw files + cloud flag). Sample: `GOESNG_L1` (symlinks under `/mnt/ceph/samples/downloads/eoread/GOESNG/`).
- The legacy function is unusable (undefined `naming`/`eo`/`flags`/`file_1km` globals) and is not a parity reference; deviations are documented in `eoread/compat.py` and the REFACTORING.md decision log.
### HYPSO — DONE (ported on user request)
- Ported: `eoread/readers/hypso.py` (`Level1_HYPSO`) for the NTNU HYPSO-1 L1B hyperspectral product: 120 bands (~388-803 nm) as uint16 radiance → canonical `Ltoa` float32 on `(bands, y, x)`, south-up; geolocation + viewing/solar angles from the legacy `navigation` group or the current `geometry` group; wavelengths from the file's own table (attribute or variable). Tests: `tests/test_hypso.py` (32 real-data tests, two chunkings, raw-file parity). Sample: `HYPSO_L1` (symlink to `vancouver_2022-07-30_1825Z-l1b.nc`).
- The legacy function is not a value reference (uint16 payload, north-up, missing bands coordinate); deviations are documented in `eoread/compat.py` and the REFACTORING.md decision log.
### NASA L1A
- Legacy: `eoread/legacy/nasa.py` — reads MODIS / VIIRS / SeaWiFS Level-1 via the L1C approach (products generated with SeaDAS l2gen; OCSSW install documented in the module docstring) + `tests/old/test_nasa.py`.
- Open questions: which of the three instruments are actually read downstream (the reader spans all three); sample availability.

**Done when:** NASA L1A — the last open item above — is agreed in a future session (scope and order). VENUS, MERIS, PACE, SGLI, GOESNG, HYPSO and the four DIMAP readers are already ported and no longer part of the open list.

---

## Cross-cutting rules (every phase)

- Real samples only; missing data ⇒ clean skip, never an error cascade.
- No structural refactors of legacy readers (stopgaps only, and only user-visible breakage).
- Every new reader: `chunks_default` class attr with today's values; `finalize()` always runs `validate()`.
- Version/variant unknowns raise at open time — never silent fallbacks.
- No-data gate (import smoke + collect-only) must stay green on every commit.
