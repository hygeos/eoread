# eoread v2 — implementation plan

Companion to `REFACTORING.md` (agreed design). This plan incorporates every decision from the 2026-10-13 design sessions. Phases are sequential; each phase ends in a commit and a green no-data gate.

## Decisions incorporated (reference)

- Lean plain Python: `Reader` base + shared transforms + `schema.validate()` — no spec engine.
- Level = class, type = variant table (metadata-detected), version = explicit if/elif + loud raise.
- Naming: level explicit — `Level1C_MSI`, `Level2A_MSI`, `Level1B_Ecostress`.
- Day one: all current reader modules move to `eoread/legacy/`; **no re-exports** (deliberate import break).
- Stopgaps on legacy code: minimal safe user-visible fixes only.
- v1-compat: one shared `eoread/compat.py::v1(ds)` over the canonical schema; pure rename/reorg = compat, value differences = documented clean break per reader.
- Phase 1 sensors and order: **MSI → ECOSTRESS → OLCI → MODIS → OLI**. Phase 2 (later): VENUS, PACE, MERIS, SGLI, NASA L1A. GOESNG stays deferred. DIMAP gets a deadline via VENUS.
- Chunk defaults: values unchanged per reader, declared as `chunks_default` class attr.

## Canonical schema (locked)

- **Registry**: `core.geo.naming.names` is the single source of truth; `eoread/schema.py::validate(ds)` checks presence/dims/dtype/units/ranges from it. New names are proposed upstream in `core` before a port needs them.
- **Dims**: `y`, `x` (pixel indices), `bands` (sensor-native string labels: `B01…`, `b01…`). One grid per dataset.
- **Multi-resolution**: single grid; `resolution` param chooses target (default = finest); native per-res output is an opt-in mode. Coarse bands upsample lazily via dask — no data lost.
- **Payload**: exactly one *primary* payload var from `{Ltoa, Rtoa, BT}`, declared per class; extra declared payload vars allowed where the product natively carries them (MODIS: `Rtoa` alongside primary `Ltoa`). Radiance products emit `Ltoa` (fixes MSI L1C misnomer); reflectance keeps `Rtoa`.
- **Geolocation**: 2D vars `latitude`, `longitude` (degrees, float32) over `(y,x)` — mandatory.
- **Angles**: `sza`, `saa`, `vza`, `vaa` — 2D full grid, degrees, float32. Raw tie-point data allowed as optional `*_tie` extras.
- **Bands**: string labels + `cwav` var (nm); optional `bands_group` coord.
- **Attrs required** (validate fails without them): `sensor`, `platform`, `product_name`, `datetime`, `input_directory`, `resolution`, `shortname`, `crs`, plus new `schema_version` (stamped by `finalize()`). Where a product lacks e.g. a CRS string, the reader derives it (e.g. from UTM zone).
- **Flags**: `_flag_reader` attr only; decode on demand. No flags/quality vars in the dataset.
- **Tests**: real samples only (no synthetic fixtures); clean skip when data/env missing; one documented data manifest.

---

## Phase 0 — skeleton + legacy move

1. **Legacy move.** `git mv` all sensor reader modules into `eoread/legacy/`; strip re-exports from `__init__.py`. `tools.py` shared helpers that the new code will use are extracted (below); legacy-only helpers move with the readers. `flags.py` stays at top level (new and legacy both import it).
2. **Stopgap.** Apply the one-line MSI `drop_vars(['y','x'])` fix in `legacy/msi.py` (cluster A, 100% broken) — the only legacy bug fix permitted in this phase.
3. **`eoread/reader.py`.** `Reader` base: fixed signature `(path, chunks=None, metadata_template=None, verbose=False)`; class attrs `sensor`, `level`, `chunks_default`, `required_vars`, `payload_var(s)`; `_finalize(ds, metadata)` = attach filtered metadata + `_flag_reader` + `schema_version` → `schema.validate(ds)`. Exceptions `UnsupportedProductVersion`, `UnsupportedProductType`. Generic variant support: optional `VARIANTS` table + `detect_variant()` (metadata-based, loud raise on unknown).
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

## Phase 2 — ECOSTRESS (`Level1B_Ecostress`)

1. Inspect the available L1B sample types; confirm the shared contract (user-confirmed: all 5 share one output); record each variant's layout diffs.
2. `_EcostressBase(Reader)`: H5 open, orbit/attitude metadata, band table.
3. `Level1B_Ecostress`: 5-entry `VARIANTS` table with metadata-field detection; loud raise on unknown type. (Port any other ECOSTRESS level currently in active use — check during the phase.)
4. Compat mapping or documented break.

**Done when:** every available L1B variant reads through the one class and passes the harness; an unknown/foreign product raises at open time; `legacy/ecostress.py` + old tests deleted.

## Phase 3 — OLCI (L1 + L2)

1. Port both levels over a shared `_OlciBase(Reader)` (HDF5 open, granule metadata); level names taken from the actual product types encountered.
2. L1 → payload `Ltoa`; L2 → primary `Rtoa`, `rho_w` kept as declared extra (today's behavior).
3. Fix the `cwvl` length mismatch (10 vs 9 bands) properly — it becomes a validate() catch, not a per-reader patch.
4. Compat mapping or documented break.

**Done when:** harness passes on local OLCI-FR (983 M) and OLCI-RR (449 M) samples; `legacy/olci.py` + old tests deleted.

## Phase 4 — MODIS (`Level1_MODIS`)

1. Engine contract with loud fallback: netCDF4 → `load_hdf4` (fixes the cluster D ImportError class).
2. Port: 36 bands, string labels, `cwav`; primary payload `Ltoa` + declared extra `Rtoa`; keep `*_tie` raw tie-point extras; single grid at target resolution (default finest = 250 m) replacing the red-channel split dims.
3. Compat mapping or documented break.

**Done when:** harness passes on the MODIS_AQUA ceph sample, including under the `load_hdf4` fallback path; `legacy/modis.py` + old tests deleted.

## Phase 5 — Landsat OLI (L8/L9)

1. Port the level(s) currently in use (check `legacy/oli.py` scope during the phase); payload `Rtoa` for reflectance products; fix the latent `cwvl` alignment issue as part of the port, not a patch.
2. Compat mapping or documented break.

**Done when:** harness passes on the LANDSAT_OLI ceph sample; root cause of today's failing `test_l1c_main` is gone by construction; `legacy/oli.py` + old tests deleted.

## Phase 6 — consolidation

1. Final pass over `compat.v1` (all five sensors) + per-reader clean-break documentation.
2. Packaging: sync `pyproject.toml` with the new layout (absorbs old TODO step 6).
3. Keep `REFACTORING.md` in sync with any decisions made during implementation (it already reflects the full 2026-10-13 design state).
4. Phase-2 prep notes: VENUS/DIMAP deadline, PACE, MERIS, SGLI, NASA L1A scoping. GOESNG remains deferred.

**Done when:** package installs and imports clean from the new layout only; `REFACTORING.md` reflects the final state; phase-2 list agreed for a future session.

---

## Cross-cutting rules (every phase)

- Real samples only; missing data ⇒ clean skip, never an error cascade.
- No structural refactors of legacy readers (stopgaps only, and only user-visible breakage).
- Every new reader: `chunks_default` class attr with today's values; `finalize()` always runs `validate()`.
- Version/variant unknowns raise at open time — never silent fallbacks.
- No-data gate (import smoke + collect-only) must stay green on every commit.
