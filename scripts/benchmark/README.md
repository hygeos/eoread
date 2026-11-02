# v1 vs v2 benchmark (time & RAM)

Compares the legacy **v1** release (master `5122d8b`, eoread 0.5.0,
function-based API) against the current **v2** dev branch (class-based readers)
across the sample products registered in `eoread/samples.py`. For each sample
it measures **wall time** and **peak RAM** at three tiers of work.

## What is measured

| tier | work performed |
|---|---|
| `probe` | build + inventory of 2D+ data variables (`name -> nbytes`), no compute |
| T0 | reader construction only — must stay lazy (no pixel compute) |
| T1 | construction + compute of a central **500×500 px** slice of the payload |
| T2 | construction + full-grid compute of the payload |

**Same-variable assertion.** For each sample the driver intersects the v1/v2
probe inventories and pins the largest common variable (by total bytes) as
`--payload`; both branches' T0/T1/T2 must carry that exact variable — asserted
inside the child (`run_one.py`) and cross-checked by the driver after every
measurement. If the intersection is empty the sample is reported *not
comparable* with its reason (probe failure, no common variable, …) — never
silently dropped.

**Fairness protocol.**

- One **fresh subprocess per measurement**: `pixi run python3 run_one.py ...`
  executed inside each branch's own checkout, so every measurement pays its
  own interpreter + import cost and no state leaks between runs.
- Metrics are read from the child: wall time (build, and build+compute for
  T1/T2) and **peak RSS** (`ru_maxrss` high-water mark, MB — includes the
  interpreter/import baseline, which is identical on both sides).
- `ROUNDS=3` by default; within each round, per sample and tier, the v1/v2
  order alternates to counter machine drift. **Medians** are reported.
- Both sides use pure defaults: v1 = path-only function call (signature
  defaults, product passed as a `pathlib.Path` — the DIMAP readers require it),
  v2 = `Class(path)` with the class's `chunks_default`.

## Setup

v2 side is this repository — make sure its environment is installed:

```bash
pixi install
```

v1 side is a throwaway **clone** of this repo, checked out at master `5122d8b`,
in the sibling directory `eoread-v1` (the driver's default — from the parent
dir: `./eoread-v1` vs `./eoread`; override with `--v1-path`). From inside the
v2 repo:

```bash
git clone "$(git rev-parse --show-toplevel)" ../eoread-v1
cd ../eoread-v1
git checkout --detach 5122d8b
pixi install
pixi add xmltodict   # scratch: v1 DIMAP readers import it but never declare it
```

Notes on that setup:

- `pixi add xmltodict` modifies the clone's `pyproject.toml`/`pixi.lock` —
  expected, it is a throwaway checkout.
- `_stubs/jrc_rayleigh_processor` satisfies the v1 module-level import of
  `jrc_rayleigh_processor.utils.iter_through_subset` (geosat / pleiades /
  planetscope). Path-only calls never reach that code path; the stub raises
  loudly if they ever do.
- Sample products must exist at the paths in `eoread/samples.py`
  (the shared `/mnt/ceph/samples` archive). Missing samples fail their probe
  with a clear error and end up in the *not comparable* section.

## Run

Via pixi tasks (from the repo root):

```bash
pixi run bench          # full run: all samples, 3 rounds (~hours; background it)
pixi run bench --rounds 1 --samples PACE_L1B,OLI_L1TP_L8   # quick smoke
pixi run bench-dry      # print the measurement plan without launching anything
```

Or directly from anywhere (`benchmark.py` is host-side, stdlib only):

```bash
python3 scripts/benchmark/benchmark.py full
python3 scripts/benchmark/benchmark.py full --rounds 1 --samples PACE_L1B,OLI_L1TP_L8
python3 scripts/benchmark/benchmark.py full --dry-run
```

Options: `--v1-path` (default: sibling `eoread-v1` directory of the v2 repo),
`--v2-path` (default: the repo containing this script), `--rounds N`,
`--slice W` (T1 slice width in px), `--samples A,B`.

## Output

Each run writes to `scripts/benchmark/results/<UTC stamp>/` (gitignored):

- `results.jsonl` — one line per measurement (probe = round 0), including
  stderr tails for failures;
- `report.md` — per-sample tables with median build/total wall time, peak RSS
  and the v2/v1 ratio per tier, plus a *not comparable* section with reasons.
  Rewritten after every round, so an interrupted run still has a report
  (marked *partial*).

## Caveats

- **v1 reader failures are kept as-is** (per agreement): they are recorded as
  FAIL rows/notes, not hidden.
- `ECOSTRESS_L1B` has no v1 reader → automatically not comparable.
- `PRISMA_L1` needs a sibling Level2C companion (per-pixel angles) that the
  archive does not hold, and v1's reader resolves the product path — so before
  probing, the driver stages a **real copy** of the L1 plus a synthesized L2C
  (constant grids; see `stage_prisma.py`) under `_staged/PRISMA_L1/`
  (gitignored). A symlink would not do: v1 would resolve it back to the
  archive and lose the companion. Staging is idempotent — an existing staged
  granule or a real L2C dropped next to it takes over.
- Comparability is decided per run by the probe: both branches must pin a
  variable with identical dims (order-insensitive) and per-dim sizes, where v2
  may read a superset of bands; rows where one branch's probe fails or no
  common 2D+ payload exists are not comparable (e.g. ECOSTRESS L1B / L2
  v001/v002, GOESNG L1).
- Peak RSS is a high-water mark: it includes everything alive at once in that
  process (construction artifacts + materialized output). T2 on large payloads
  (PACE, MSI res-10) genuinely materializes gigabytes.
- Page-cache warmth differs between rounds; medians over interleaved rounds
  mitigate this. Run the benchmark when the machine is quiet.
- Timeouts per tier: probe 600 s, T0 900 s, T1 900 s, T2 3600 s.
