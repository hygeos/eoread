# eoread refactor — step-by-step plan

User-driven: each step is discussed, agreed, then applied on its own.
Line numbers verified 2026-10-13 (post step 1).
Steps 2–7 reordered 2026-10-13 (user decision):
**bugs → docstrings → chunking → tests → packaging → DIMAP readers + housekeeping**.
Tests come after the behavior-changing steps so they are rewritten against the
final API once, not twice.

## Step 1 — Remove dead / deprecated code ✅ DONE

**Removed (8 modules, −1100 lines):**
`eoread/eo.py`, `common.py`, `utils.py`, `ancillary_nasa.py`, `era5.py`, `gsw.py`,
`dem.py` (incl. unreachable `SRTM()` body + `_ArrayLike_SRTM`/`read_hgt`),
`download_eumdac.py`.

**Unblocked before removing `eo.py`:**
- `olci.py` — `raiseflag` now from `core.tools` (top import); dead local import removed.
- `nasa.py` — `scattering_angle`/`init_geometry` moved in as `_scattering_angle`/`_init_geometry`;
  call sites use direct `core.tools` imports (`merge`, `raiseflag`).

**Verified:** no leftover references (except accepted leftovers below), syntax OK,
all live readers import in the pixi env.

**Kept by explicit decision (do not touch):**
- `eoread/goesng.py` + `tests/test_goesng.py` — fully broken but kept on purpose.
- `eoread/__pycache__/` (~132 stale .pyc), Makefile `samples` target,
  commented-out entries in `scripts/eval_readers.py`.

**Accepted leftovers (handled in later steps):**
- `tests/test_msi.py:55` — skipped test uses unimported `eo` → step 5.
- `tests/test_goesng.py:83,88` — undefined `eo.init_Rtoa` → stays with kept goesng.

> Step 1 changes are committed as `cfa7370` ("Remove dead code").

## Step 2 — Fix real bugs in live readers

- `venus.py:460` `supplement_latlon`:
  - builds lat/lon with a plain linspace over UTM corner values (no projection applied);
  - row 0 gets the *minimum* latitude → violates top-to-bottom convention;
  - key path mismatch: uses `Geopositioning/Global_Geopositioning` (:468) while
    `read_metadata` uses `Geoposition_Informations/Geopositioning/Group_Geopositioning_List` (:441-443).
- `nasa.py:328` — bare `except: pass` around the per-band rtoa computation swallows real errors.
- `oli.py:43` — `cwvl` has 10 values but `band_names` (:30) has 9; the extra thermal
  entries (10895, 12050 nm) make the `cwav` assignment at :134 dimension-inconsistent.
- The `_v1_compat` mess:
  - `meris.py:158` — calls `_v1_compat(ds, prod, lock, chunks)`; no such function
    exists in the module → guaranteed `NameError` when `v1_compat=True`.
  - `ecostress.py:112` — calls `_v1_compat(l1)`; not defined there either
    (only `olci`/`msi`/`hypso` each define their own local copy).
  - `modis.py` / `venus.py` — accept a `v1_compat` parameter but never use it → silent no-op.
  - Decision: consolidate into one shared helper (olci/msi/hypso bodies are similar);
    drop the param from modis/venus if their v1 format doesn't make sense —
    confirm during implementation.
- `planetscope.py:273` — `_srf_getter` points at `read_srf_superdove` for a PlanetScope
  product (wrong sensor), and no `_srf_getter_arg`. Decision: delete the attr
  (no correct PlanetScope SRF source exists).
- `planetscope.py:264` — stray no-op statement `ds.attrs["acquisition_date"]`.
- `oli.py:103` — dead `if isinstance(chunks, int)` after `format_chunks`
  already returns a dict.
- `modis.py` — no-op loop `for v in uncert_vars: l1 = l1.assign()`; dead module-level
  Planck constants (duplicated inside `compute_bt`); typo "Revize variables" (:82).

(Original item "gsw 10° lat-coord shift" is moot — `gsw.py` was deleted in step 1.)

**Acceptance:** venus lat/lon consistent with metadata keys and row order;
no bare excepts in `nasa.py`; oli band/cwvl lists aligned; every reader exposing
`v1_compat` either has a working path or loses the param; import smoke-check +
`pytest --collect-only` still clean.

## Step 3 — Docstring pass

- `ecostress.py` — example says `Level1_ECOSTRESS`; actual class is `Level1C_ECOSTRESS`.
- `sgli.py` — module docstring says "ECOSTRESS H5file" (wrong product).
- `hypso.py` — "NTNU HYPSO-1" naming; missing `v1_compat` in Parameters.
- `meris.py` — claims `dir_smile` defaults to `'../auxdata/meris/'`; actual default is `None`.
- `venus.get_sample` — claims `ValueError` on bad level (no such check exists);
  typo "does not exists".
- `modis.py` — docstring mentions float band ids 13.5/14.5.
- `README.md` — references deprecated `core.to_netcdf` and eo-based subsetting
  (`eo` no longer exists after step 1 → definitely stale).

Note: chunk defaults changed in step 4 will re-touch some lines written here;
that is covered by step 4's acceptance.

**Acceptance:** every class name, default value, and example in docstrings matches code;
README examples runnable as written.

## Step 4 — Chunking consistency

- Inconsistent defaults: `500` (most readers), `100` (`modis`), `1000` (`goesng`),
  `{"x": 1024, "y": 1024}` (`geosat`, `planetscope.py:75`) — pick one convention.
- DIMAP readers bypass `eoread.tools.format_chunks` (final handling in step 7).
- `nasa.py` — scalar int chunks get applied to every dim *including* bands;
  route through `format_chunks`.
- `venus.py` — 24 sequential per-band zip-mask opens by default; `CLA_ALL`
  resampled with `'repeat'` (silently duplicates pixels).

**Acceptance:** one chunk convention across all live readers; docstrings affected
by the change are updated in the same step; import smoke-check +
`pytest --collect-only` still clean.

## Step 5 — Fix broken tests (non-goesng)

Rewritten against the *final* API (post steps 2–4) so tests target settled
behavior and chunking defaults once, not twice.

- `tests/test_hypso.py` — calls nonexistent module-level functions
  `generic.test_main` (:30), `test_execution_time` (:34), `test_lazy_load` (:43),
  `test_read` (:51), `test_subset` (:55), `test_plot` (:58).
  Rewrite against the real API: `generic.Test.main/execution_time/lazy_load/read/subset`
  + module-level `generic.plot`; use `tests/test_msi.py` as the reference pattern.
  Also verify `generic.compare_version` (:40) exists in `tests/generic.py`.
- `tests/test_olci.py:20,27` — `test_plot` defined twice (second silently shadows first).
  Decide which body is correct, delete the other.
- Skipped `test_read` tests (e.g. `tests/test_msi.py:49-56`, check others) reference
  nonexistent `param`/`indices` fixtures and unimported `eo` — rewrite or drop.
- Normalize fixture scopes (`module` vs `session`) so sample products are
  downloaded once per run, not per test module.
- `tests/test_pace.py::test_main` calls `.compute()` before the lazy-load assertions,
  making them vacuous — fix the test.

**Acceptance:** `pytest --collect-only` clean; no duplicate test names;
no references to nonexistent `generic.*` symbols or fixtures.

## Step 6 — Packaging sync (`pyproject.toml`)

- `[project.dependencies]` missing runtime deps actually imported by the package:
  `xarray`, `dask`, `numpy`, `rasterio`, `pandas`, `pyhdf`, `scipy`, `python-dateutil`,
  `xmltodict`, `jrc_rayleigh_processor`, `cdsapi`, `eumdac`.
  Declare everything currently imported — incl. `xmltodict` / `jrc_rayleigh_processor`
  for the DIMAP readers kept as-is for now; small follow-up diff in step 7
  only if that step changes imports.
- `requires-python = "< 3.14"` while cpython-3.14 artifacts exist — decide: allow 3.14
  or keep the pin (stale `__pycache__` is kept by decision, but the pin should match intent).
- No `eoread/__init__.py` (implicit namespace package) — confirm that's intentional
  and document it, or add the file.

## Step 7 — DIMAP WIP readers: align, dedupe, or mark experimental (+ final housekeeping)

Scope: `geosat.py`, `planetscope.py`, `pleiades.py`, `pleiades_neo.py`.
Safe to run last: no tests cover these modules and they are import-isolated.

- Deduplicate the ~4 near-identical `parse_dimap_*` parsers into one shared helper
  (candidate home: `eoread/tools.py`).
- `planetscope.py` has two XML paths: hand-rolled `xml_to_dict`/`parse_planet_xml`
  (used at :117) and `xmltodict.parse` (:269) — pick one.
- `geosat.py:299` — `final_ds[str(v)].attrs = v.attrs` mixes enum members into attrs; fix.
- Convention gap vs the new-style readers: old `core.geo.n` naming, `x`/`y` dims instead
  of rows/columns, no flags/SRF handling, bypass `format_chunks`.
  Decide per reader: align to new conventions **or** explicitly mark experimental
  (docstring + README).
- If imports change as a result, sync `pyproject.toml` deps (step 6 follow-up).

**Housekeeping:** refresh stale notes in this file; commit `TODO.md`.

---

### Out of scope / parked
- `goesng.py` + its test: broken, kept by decision — no fixes planned.
- Stale `eoread/__pycache__/`, Makefile `samples` target, `eval_readers.py` comments:
  kept as-is by decision.
