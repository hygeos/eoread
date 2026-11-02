# Test data policy

Tests run on **real products only** — there is no synthetic fixture factory.
Each test that needs data resolves a path through the sample-product registry
(`eoread/samples.py`) via `get_sample()` (`tests/conftest.py`): pass a reader
class (its `sample_products` IDs are tried in order) or a sample ID directly.
If no candidate resolves, the test is **skipped**, never errored: a machine
without samples must produce clean skips, and the no-data gate must stay green
on every commit.

## Opt-in provider downloads

A missing local path does not always mean "unavailable". Samples registered
with a `Download(provider, product_id)` spec are fetched on demand through
SAND into the local cache (`~/.cache/eoread/samples`, flat layout
`<cache>/<product_id>`, override with `EOREAD_SAMPLES_DIR`) and resolved from
there. The opt-in list is the registry itself — currently `MSI_L1C`,
`MSI_L2A` (cdse), `ECOSTRESS_L1C` (nasa) and `VENUS_L1C` / `VENUS_L2A`
(cnes); making a new sensor downloadable is one line per product in the
registry. The two VENUS entries are currently blocked by a known CNES STAC
API query issue — they fail with the same clean skip until fixed upstream.
Any download failure (missing `samples` extra, no credentials, network
error) degrades to the same clean skip as a missing local path; provider
credentials (when required) live in `~/.netrc`.

Where a downloadable granule differs from the shared-archive copy (MSI L1C,
ECOSTRESS L1C, both VENUS levels), the tests are written granule-agnostic —
no pinned tiles, dates, CRS zones or grid sizes.

## Layout

- `tests/` — v2 conformance tests.
- `conftest.py` and `generic.py` are shared infrastructure at the root.

## Sample locations

The registry in `eoread/samples.py` is the single source of truth for sample
paths; each reader class declares which IDs it accepts via `sample_products`.
Sources:

- `data/sample_products/` — bundled products (repo-relative paths).
- `/mnt/ceph/...` — shared archive, absolute paths (developer machines).
- `~/.cache/eoread/samples` — SAND downloads of the opt-in list above.

## No-data gate

```bash
pixi run -e default gate        # import smoke-check of all modules + pytest --collect-only
```

Must be green on every commit; it runs without any sample data and without
the `samples` extra.
