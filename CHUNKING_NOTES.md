# Chunking notes — MSI reader (parked for later)

Working notes from the chunking investigation (Oct 2026). All numbers measured
on the sample S2A T23KPQ L1C, band B04, res-10 grid 10980×10980 uint16
(122 MB on disk, 241 MB raw), stored JP2 tiles **1024×1024**.
Environment: python 3.13, numpy 2.3.5, dask 2026.1.2, rasterio 1.5.0.

Reproducer for the graph-cost part: `scripts/check_chunk_open.py`
(`pixi run python scripts/check_chunk_open.py [product]`).

## Settled questions

### 1. Is a post-read "rechunk everything" step worth it? — No

- Our pipeline chunks at open time and every transform (`spatial_resample`,
  `latlon_utm`, `tiepoint_angles`) emits its output directly in the target
  chunk spec; all variables share identical chunk boundaries. The graph *is*
  the final layout, so a blanket `ds.chunk(spec)` after reading only adds an
  extra full repartitioning pass (~13 GB at res-10) for zero gain.
- Users who need a different block size should rechunk lazily, per variable,
  inside the same compute chain (before `.compute()`), never on materialized
  data.

### 2. Does specifying `chunks=` at open time add graph complexity? — No

Measured with several user specs (Part 1 of the script):

| chunks spec            | blocks | nodes  | layers | embedded numpy | build RSS |
|------------------------|-------:|-------:|-------:|---------------:|----------:|
| (500, 500)             |    484 |    969 |      3 |           0 MB | flat      |
| (333, 777), odd        |    495 |    991 |      3 |           0 MB | flat      |
| (1024, 1024) tile-al.  |    121 |    243 |      3 |           0 MB | flat      |
| (10980, 10980) whole   |      1 |      3 |      3 |           0 MB | flat      |

Graph cost is exactly `2×blocks + 1` nodes for any spec — the minimum for a
chunked representation. Nothing materialized at build time. **Keep the
`chunks=` knob on the reader** (`Level1C_MSI(path, chunks=…)`).

Also settled: rioxarray's unchunked open is **eager** (returns a fully
decoded numpy array — 482 MB float32 for this band, +744 MB RSS at build).
That is why `_read_bands` must always pass `chunks=`.

## Open question (to tackle later)

### First-compute wall time: 500-px chunks vs tile-aligned 1024-px chunks

Decode breakdown measured by spying on `DatasetReader.read` during compute:

- A chunked-open compute performs **exactly 484 windowed reads** (one 500×500
  `read(window=…, masked=True)` per dask block) — no hidden full-array read.
- Compute #1 (fresh open): **2.28 s**. Compute #2 on the same array/handle:
  **0.36 s** with identical call count → the open dataset keeps decoded tiles
  in memory for its lifetime; the expensive part is a one-time decode.
- Threaded scheduler gives no speedup (num_workers=1: 2.20 s ≈ default
  2.26 s): reads are serialized on the shared handle.
- The 2.28 s vs 0.16 s (eager full-array decode) gap comes from:
  1. **tile over-read amplification** — a 500 px window straddles up to a 2×2
     neighborhood of the stored 1024 px tiles, so each tile is decoded once
     per adjacent window (up to ~4×) instead of exactly once;
  2. **per-window overhead** — 484 GDAL call crossings + per-window mask
     construction instead of one contiguous pass.

Honest trade-off (warm page cache, one band):

|                        | A: chunks=500 at open      | B: unchunked (eager)          |
|------------------------|----------------------------|-------------------------------|
| open                   | +0 s, +0 MB                | 0.16 s, **+744 MB**           |
| full compute           | **2.3 s** (one-time decode)| 0.1 s                         |
| 2nd compute            | 0.36 s (handle tile cache) | 0.1 s                         |

Eager wins on wall time *only* if the whole band is materialized anyway and
RAM allows — which it does not in our case (×13 bands at res-10 ≈ 10 GB
resident; the machine OOM-kills well below that). Lazy stays the right
default for memory and partial computes.

**Candidate fix to close most of the gap while staying lazy:** open with
`chunks=1024` (tile-aligned) — each window then maps to exactly one stored
tile (amplification → ~1×) and 484 reads drop to ~121, which should approach
eager's decode speed with bounded memory.

### TODO

- [ ] Benchmark `chunks=1024` vs `chunks=500`: first-compute wall time + peak
      RSS, one band, res-10 (warm and cold page cache if possible).
- [ ] Decide: keep `chunks_default = 500` (legacy parity) and document 1024 as
      the recommended spec, or change the default.
- [ ] If benchmarked: add a decision-log row to `REFACTORING.md`.
