# eoread refactoring — agreed design approach

**Status:** design fully agreed (sessions of 2026-10-13). All former open items are resolved; the execution plan is `IMPLEMENTATION_PLAN.md`.
**Companions:** `IMPLEMENTATION_PLAN.md` (phased execution plan) · `TODO.md` (incremental fixes of the current code — largely superseded, see §9) · `TESTS_PROBLEMS.md` (current defect catalog, evidence base for this design).

---

## 1. The role of eoread (unchanged target)

Unified ingestion layer: turns raw, vendor-specific satellite products (Level-1/Level-2) into a
single, homogeneous, **lazy** `xarray.Dataset`, so everything downstream is sensor-agnostic.

Stack position: raw vendor files → **eoread** → standardized xarray datasets → `core`/`hygeos` processing.

**Core responsibilities**
- Format bridging — one call per product type regardless of underlying format (SAFE+GeoTIFF+XML, HDF4/5, NetCDF, DIMAP, binary+sidecar).
- Standardized data model — canonical dims (`bands`, `y`/`x`) and variables (payload radiance/reflectance/brightness-temperature, `cwav`, sun/view angles on the full grid, 2D lat/lon rasters), consistent attribute conventions. The locked contract is §5.
- Geometry at read time — UTM→lat/lon, tie-point angle interpolation to full resolution.
- Lazy, chunked I/O — dask-backed; reading must not compute.
- Metadata curation — granule/product metadata into dataset attributes, filterable via `metadata_template`.
- Flag decoding — flags framework turning packed flags into readable boolean masks, **on demand** (`_flag_reader` attr).

**Explicit non-goals**
- No radiometric/atmospheric processing, no resampling to user grids, no storage format of its own (a *view*, not a converter). Normalizing a multi-resolution product onto one target grid at load time is internal consistency work, not "resampling to a user grid".
- No product discovery/catalog API — the caller hands it a path.
- No downloading: sample-data plumbing is test support, not library code.

## 2. The difficulties the design must solve

| # | Difficulty | Evidence in current eoread |
|---|------------|----------------------------|
| 1 | Input-side heterogeneity (every sensor = format archaeology) | ~20 bespoke 300–700-line reader modules; DIMAP stalled as WIP |
| 2 | Canonical model is a lossy contract that calcifies | olci `cwvl` length mismatch (10 vs 9 bands) undetected; int/str band-label mix |
| 3 | Product versions drift silently | `_v1_compat` bolted on as local copies in 3–4 readers, 2 now NameErrors |
| 4 | Geometry bugs are silent failures | venus `supplement_latlon`, msi interp/`drop_vars` breakage, ecostress lat ordering |
| 5 | Lazy I/O easy to promise, hard to keep honest | `.compute()` on 1.6 GB in tests; chunk defaults drifted (500/100/1000/1024) |
| 6 | Testing needs data you can't always have | 118 failing tests, most from missing data/env, not code |
| 7 | Dependency fragility, opaque C-level errors | modis ImportError with a working `load_hdf4` next door; xarray drift broke msi; flaky HDF5 under xdist |
| 8 | API uniformity decays by copy-paste | the whole step 1–3 audit: duplicated helpers drifting per module |
| 9 | Scope creep blurs the role | OCSSW `make_L1C`, SAND downloads, plot helpers living inside the reader library |

## 3. Design principles (agreed)

1. **Lean plain Python — no declarative spec engine.** A configuration layer (spec files,
   path-expression language, interpreter) was considered and **rejected** as over-engineered for
   ~20 readers / one team ("usine à gaz"): extra machinery to build and debug through, indirection
   in every stack trace. Uniformity comes from a base class + shared functions + conformance tests,
   not from a schema interpreter.
2. **One `Reader` base class, fixed signature** — `path, chunks=None, metadata_template=None, verbose=False`
   plus a few shared helpers (granule discovery, base attributes, finalize). A **conformance test**
   ("every registered reader instantiates and passes `validate()` on its sample") enforces uniformity.
3. **Shared transforms module — one implementation per operation.** `latlon_utm`, `tiepoint_angles`,
   band resampling/radiometry helpers live in one place and are *called* by readers like any
   function. Per-reader copies are deleted, not synchronized. This structurally kills difficulty 4.
4. **Schema validation at the end of every read.** A single `validate(ds)` checks dims/vars/dtypes/units/ranges
   (never values) against the **`core.geo.naming.names` registry** — the single source of truth for
   canonical names — plus the class's declared `required_vars` and the mandatory attribute set (§5).
   Called in `finalize()`. Contract violations become loud read-time errors (difficulty 2). New
   canonical names are proposed upstream in `core` before a port needs them.
5. **Versions and variants: detect from metadata, diff-table, loud raise on unknown.** Never silent
   misread. Detection reads a field *inside* the file (header/product-type), not filename regexes
   (difficulty 3).
6. **Level = class, type = variant table** (see §6). The public API unit is (sensor, level) because
   that is the unit of the output contract.
7. **Chunk defaults unchanged in value** — kept as today per reader for performance — but declared
   as a visible `chunks_default` class attribute instead of buried signature defaults; single chunk
   policy still lives in `format_chunks` (difficulty 5). Snapping to on-disk chunks is *not* the
   default; it may exist later as an opt-in.
8. **Real-data-only test policy.** No synthetic fixture factory. Samples are real products; the
   problems to fix are *availability and failure hygiene*, not speed (full suite runs ~2 min)
   (difficulty 6, see §7).
9. **Scope split.** Core keeps only readers + schema + transforms + flags. `make_L1C`/OCSSW and
   sample downloaders leave the library (difficulty 9, see §8).

## 4. Architecture sketch

Module layout:

```
eoread/
  reader.py        # Reader base class, registry, finalize/validate hook
  schema.py        # validate(ds) against core.geo.naming.names + per-class contracts
  compat.py        # v1(ds): the single shared v1-compat transform
  transforms/      # latlon_utm, tiepoint_angles, resample, radiometry — shared
  flags.py         # flags framework (as today)
  readers/
    msi.py         # _MSIBase + Level1C_MSI + Level2A_MSI
    ecostress.py   # _EcostressBase + Level1B_Ecostress (+ variant table)
    ...one module per sensor
  legacy/          # ALL current reader modules, moved here day one; deleted per sensor as ports land
```

### Reader contract

Every reader class declares: `sensor`, `level`, `chunks_default`, `required_vars`, payload var(s);
implements `read(path) -> xr.Dataset`; ends with `self.finalize(ds, metadata)` which attaches
filtered metadata + `_flag_reader` attr + `schema_version` attr and runs `validate(ds)`.

### Example — MSI (levels share a sensor base)

Today `Level2_MSI` is a ~150-line near-clone of `Level1_MSI`. Target:

```python
class _MSIBase(Reader):
    def read(self, dirname):
        ...shared skeleton: granule discovery, XML parsing, base attrs,
            band table, latlon_utm(...), tiepoint_angles(...) ...
        ds = self.payload(ds, granule, root)      # the only level-specific step
        return self.finalize(ds, root, gran)

class Level1C_MSI(_MSIBase):
    sensor, level = 'MSI', 'L1C'
    root_xml = 'MTD_MSIL1C.xml'
    chunks_default = 500                          # today's default, unchanged
    payload_var = 'Ltoa'                          # L1C is RADIANCE — the v1 misnomer is fixed
    required_vars = ['Ltoa', 'sza', 'saa', 'vza', 'vaa', 'latitude', 'longitude']
    def payload(self, ds, granule, root): ...     # TOA bands: open, (x+off)/quant, resample

class Level2A_MSI(_MSIBase):
    sensor, level = 'MSI', 'L2A'
    root_xml = 'MTD_MSIL2A.xml'
    drop_bands = ('B8', 'B10')
    payload_var = 'Rtoa'                          # reflectance keeps the ecosystem name
    required_vars = [...]                         # its own contract
    def payload(self, ds, granule, root): ...     # BOA bands
```

### Example — version handling (explicit, exhaustive)

```python
baseline = float(root['...']['PROCESSING_BASELINE'])
if baseline < 4.0:
    offset = None                    # old format: no radiometric offset
elif baseline < 5.0:
    offset = read_offset(root)
else:
    raise UnsupportedProductVersion(baseline)    # loud — today this silently does offset=0
```

### Example — variants within a level (ECOSTRESS, 5 L1B types)

One class handles all 5 because they promise the same output; the table overrides *where things
are and what they're called* only:

```python
class Level1B_Ecostress(_EcostressBase):
    sensor, level = 'ECOSTRESS', 'L1B'
    VARIANTS = {
        'type_A': dict(detect={'field': '/.../ProductType', 'value': '...'},
                       groups=dict(radiance='/Science/...')),
        # ...5 entries, a few lines each
    }

    def read(self, path):
        tree = open_header(path)
        vname = self.detect_variant(tree)
        if vname is None:
            raise UnsupportedProductType(...)     # loud, never silent
        ds = super().read(path, variant=self.VARIANTS[vname])
        return self.finalize(ds, tree)
```

## 5. Canonical schema (locked)

| Element | Contract |
|---|---|
| **Registry** | `core.geo.naming.names` is the single source of truth; `eoread/schema.py::validate(ds)` checks presence/dims/dtype/units/ranges from it. New names are proposed upstream in `core` before a port needs them |
| **Dims** | `y`, `x` (pixel indices), `bands` (sensor-native string labels: `B01…`, `b01…`). One grid per dataset |
| **Multi-resolution** | Single grid; `resolution` param chooses target (default = finest); native per-res output is an opt-in mode. Coarse bands upsample lazily via dask — no data lost |
| **Payload** | Exactly one *primary* payload var from `{Ltoa, Rtoa, BT}`, declared per class; extra declared payload vars allowed where the product natively carries them (MODIS: `Rtoa` alongside primary `Ltoa`). Radiance products emit `Ltoa` (fixes the MSI L1C misnomer); reflectance keeps `Rtoa` |
| **Geolocation** | 2D vars `latitude`, `longitude` (degrees, float32) over `(y,x)` — mandatory; dim-rename and `y_`/`x_` deviants die with their ports |
| **Angles** | `sza`, `saa`, `vza`, `vaa` — 2D full grid, degrees, float32. Raw tie-point data allowed as optional `*_tie` extras |
| **Bands** | String labels + `cwav` var (nm); optional `bands_group` coord. The int/str mix ends |
| **Attrs required** (validate fails without them) | `sensor`, `platform` (string), `product_name`, `datetime`, `input_directory`, `resolution`, `shortname`, `crs`, plus new `schema_version` (stamped by `finalize()`). Where a product lacks e.g. a CRS string, the reader derives it (e.g. from the UTM zone) |
| **Flags** | `_flag_reader` attr only; decode on demand. No flags/quality vars in the dataset |

## 6. Product levels & types — agreed rules

| Axis | Example | Changes output contract? | Mechanism |
|---|---|---|---|
| **Level** | MSI L1C vs L2A | Yes | One class per (sensor, level); shared code via sensor base |
| **Type/variant within a level** | ECOSTRESS 5× L1B | No — same vars out, different layout in | Variant table inside the class |
| **Version/baseline within a type** | S2 baseline 3 vs 4 | No — small diffs | Explicit if/elif + loud raise |

Rules:
- **(sensor, level) is the public API unit** — it is the unit of the output contract.
- A **variant may only override locations and names**. If a "type" needs different variables or
  math, it is a new (sensor, level) class, not a variant entry.
- **Direct call = verification, not guessing**: `Level1C_MSI(path)` checks the product-type field
  and raises on mismatch; no detection needed.
- **`autodetect` dispatches to a contract**, never to a layout detail: all ECOSTRESS L1B flavors
  resolve to the same `Level1B_Ecostress`, which sorts out the variant internally.
- **Only implement levels actually used**; adding one later is a thin subclass + its
  `required_vars`.
- **Naming**: level explicit, keeping the existing shape — `Level1C_MSI`, `Level2A_MSI`,
  `Level1B_Ecostress` (removes the "what does Level1 mean?" ambiguity of v1 names).

## 7. Testing policy (agreed)

- **Real samples only** — no synthetic fixtures. Speed is not the constraint (~2 min full run);
  availability is.
- **Skip guards**: every sample-dependent test skips cleanly when its product/env var is missing —
  one `SKIPPED` line, never a fixture-error cascade (the current 48-error cascades).
- **One documented data manifest** (`tests/README.md`): required env vars, expected sample
  locations, what each suite part needs.
- **Generic harness gains invariants** run on every reader: geometry checks (lat/lon
  monotonicity/ordering, angle ranges, latlon-vs-projection consistency), the laziness invariant
  (instantiation computes nothing — dask graph intact), and `validate()` conformance.
- **CI matrix** over supported xarray/dask/netcdf versions running the suite — dependency drift
  (the msi `drop_vars` breakage) caught at PR time, not in user environments.
- **No-data gate** stays: import smoke-check of all reader modules + `pytest --collect-only`.

## 8. Out of core scope (relocated)

| Item | Destination |
|---|---|
| `make_L1C` / OCSSW (docker/shell) | Out of the library — processing tool, not ingestion (hygeos or CLI with optional deps) |
| Sample download (`collect_sample`/SAND) | Test support (`tests/conftest`) or a separate extra; the library never downloads |
| Plot helpers | `core.tests` (already mostly there) |
| `autodetect` | **Kept**, reimplemented as registry dispatch over the Reader base — and the registry gives "list supported sensors + versions" for free |

## 9. Migration & coexistence (agreed)

- **Phased coexistence.** New architecture at the top level; all current reader modules are
  `git mv`'d to `eoread/legacy/` **day one**.
- **No re-exports** — upgrading is a deliberate import update, not a silent behavior change.
- **Stopgaps on legacy code: minimal safe user-visible fixes only** (e.g. the one-line MSI
  `drop_vars` fix). No structural refactors of readers that are being ported.
- **v1-compat**: one shared `eoread/compat.py::v1(ds)` transform over the canonical schema —
  since all v2 readers emit the same contract, ONE transform serves all sensors (replacing the
  4 per-reader `_v1_compat` copies). Rule: compat is supported when the v1→v2 difference is a
  **pure rename/reorganization** of variables and dims; if v1 output had different *values* for a
  sensor, that reader gets a **documented clean break**.
- **Phase 1 sensors & order**: MSI → ECOSTRESS → OLCI → MODIS → OLI (the five that carry ~90% of
  real usage; MSI first — most broken and exercises the most shared transforms; ECOSTRESS second —
  validates the variant-table mechanism early). Phase 2 (later): VENUS, PACE, MERIS, SGLI, NASA L1A.
  GOESNG stays deferred by explicit decision. DIMAP gets a concrete deadline via the VENUS port.
- **Relation to `TODO.md`**: steps 5–7 (tests, packaging, DIMAP) are absorbed into the v2 plan;
  bug-fix items for phase-1 sensors are superseded by their ports (stopgaps only in the meantime);
  `goesng.py` + its tests remain broken by explicit decision.

## 10. Decision log

| Decision | Outcome | Rationale |
|---|---|---|
| Declarative spec engine vs plain Python | **Plain Python** (base class + shared transforms + validation) | Spec layer rejected as over-engineered for ~20 readers / one team; debugging indirection not worth it |
| Output contract enforcement | `validate(ds)` in every `finalize()` + per-class `required_vars`, anchored on the `core.geo.naming.names` registry | Contract violations must be loud at read time (difficulty 2); downstream already owns the naming — don't fork it |
| Version/variant handling | Metadata-based detection, diff tables, loud raise on unknown | Kills silent misreads (difficulty 3); per-sensor content, generic discipline |
| Level/type organization | Level = class; type = variant table; sensor base shares code across levels | (sensor, level) is the output-contract unit; deletes the MSI L1C/L2A clone pattern |
| Chunk defaults | **Values unchanged** (performance), declared as `chunks_default` class attribute | User requirement; visibility/auditability without behavior change |
| Test data | **Real samples only**, skip guards + data manifest | User preference; suite speed acceptable, availability is the real problem |
| Scope | make_L1C and sample downloaders out of core | Keeps the ingestion role clean (difficulty 9) |
| Naming convention | Level explicit, existing shape: `Level1C_MSI`, `Level2A_MSI`, `Level1B_Ecostress` | Removes "what does Level1 mean?" ambiguity with minimal shape change |
| Migration strategy | Phased coexistence; legacy modules → `eoread/legacy/` day one; **no re-exports** | Clean new top level from the start; upgrading is a deliberate migration, not silent drift |
| Legacy stopgaps | Minimal safe user-visible fixes only | Users of broken sensors (MSI) aren't held hostage during the transition; no throwaway refactors |
| v1 backward compat | One shared `compat.v1(ds)` transform; pure rename/reorg = compat, value differences = documented clean break per reader | Replaces the 4 drifting per-reader `_v1_compat` copies with one tested transform over the canonical schema |
| Phase 1 sensors | MSI, ECOSTRESS, OLCI, MODIS, Landsat OLI — order MSI → ECOSTRESS → OLCI → MODIS → OLI | ~90% of real usage; de-risks shared transforms and the variant mechanism early |
| ECOSTRESS L1B types | All 5 share one output contract → one class + 5-entry variant table | User-confirmed; keeps (sensor, level) as the API unit |
| Multi-resolution products | Single grid; `resolution` param (default finest); native per-res output opt-in | One homogeneous dataset per granule; lazy dask upsampling loses no data |
| Payload naming semantics | Radiance → `Ltoa` (fixes MSI L1C misnomer); reflectance keeps `Rtoa`; no new `rboa` | Semantically correct without a large downstream blast radius; compat covers old names |
| Required attrs | Strict full set: sensor, platform, product_name, datetime, input_directory, resolution, shortname, crs + new `schema_version` | Downstream relies on these; strictness catches missing curation at read time |
| Flags in dataset | `_flag_reader` attr only, decode on demand | Dataset stays minimal and lazy-honest; no per-reader "quality" definition to standardize |
