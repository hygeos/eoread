#!/usr/bin/env python3
"""
Verify that specifying chunks AT OPEN TIME adds no dask-graph complexity.

The v2 readers open every JP2 band with an explicit chunk spec
(`open_dataarray(..., chunks=...)`), which is also the user-facing knob
(`Level1C_MSI(path, chunks=...)`). This script measures, on one real band,
that this design is graph-cheap and compares it against the alternative of
opening unchunked and calling `.chunk()` afterwards:

Part 1 — chunked open with several user specs (default, odd non-dividing,
         tile-aligned, whole-array): graph layers/nodes vs expected block
         count, largest numpy array embedded in the graph, build-time RSS.
Part 2 — A = chunks at open  vs  B = unchunked open + .chunk(): same
         metrics, plus full-array compute (wall time, RSS) and bit-exactness,
         plus a single-block compute to confirm laziness is preserved.

Expected outcome: A's graph cost is exactly ~2 nodes per requested block with
zero materialized data at build time; B eagerly decodes the whole raster into
RAM before any compute (the regression the chunked open avoids).

Usage:
    pixi run python scripts/check_chunk_open.py              # registered MSI_L1C sample
    pixi run python scripts/check_chunk_open.py <product>    # any .SAFE path
"""
import gc
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
from xarray import open_dataarray


def rss_mb() -> float:
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return -1.0


def graph_metrics(arr) -> tuple[int, int, int]:
    """(n_layers, n_nodes, largest numpy array embedded in the graph, bytes)."""
    g = arr.data.__dask_graph__()
    layers = list(g.layers.values())
    nodes, max_np = 0, 0
    for layer in layers:
        try:
            vals = list(layer.values())
        except Exception:
            continue
        nodes += len(vals)
        for v in vals:
            if isinstance(v, np.ndarray):
                max_np = max(max_np, v.nbytes)
    return len(layers), nodes, max_np


def open_chunked(jp2: Path, cy: int, cx: int):
    with warnings.catch_warnings():
        # 500 px windows do not align with the stored JP2 tiles; the reader
        # suppresses this warning at its own open call.
        warnings.simplefilter('ignore')
        if cy is None and cx is None:
            return open_dataarray(jp2, engine='rasterio').squeeze()
        return open_dataarray(jp2, engine='rasterio', chunks={'y': cy, 'x': cx}).squeeze()


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        from eoread.readers.msi import Level1C_MSI

        try:
            root = Level1C_MSI.get_sample()
        except FileNotFoundError as exc:
            print(f'SKIP — no MSI_L1C sample available ({exc}); '
                  'pass a .SAFE path as an argument')
            return 0

    jp2s = sorted(root.glob('**/IMG_DATA/*_B04.jp2'))
    if not jp2s:
        print(f'SKIP — no *_B04.jp2 found under {root}')
        return 0
    jp2 = jp2s[0]

    # Shape via rasterio: an unchunked xarray open would decode the whole
    # raster eagerly — precisely what Part 2 exposes.
    with rasterio.open(jp2) as ds:
        ny, nx = ds.shape
    print(f'file {jp2.name}: {ny}x{nx} (~{ny * nx * 2 / 1e6:.0f} MB raw)')
    print()

    print('=== Part 1: chunks specified at open time, various user specs ===')
    for cy, cx in [(500, 500), (333, 777), (1024, 1024), (ny, nx)]:
        r0 = rss_mb()
        arr = open_chunked(jp2, cy, cx)
        delta = rss_mb() - r0
        nl, nn, maxnp = graph_metrics(arr)
        blocks = -(-ny // cy) * -(-nx // cx)
        print(f'chunks=({cy},{cx}): layers={nl} nodes={nn:>5} '
              f'blocks={blocks:<5} embedded_np={maxnp / 1e6:6.1f}MB '
              f'build_rss=+{delta:5.0f}MB')
        del arr
    gc.collect()

    print()
    print('=== Part 2: A = chunks at open   vs   B = unchunked open + .chunk() ===')
    r0 = rss_mb()
    a = open_chunked(jp2, 500, 500)
    la, na, maxna = graph_metrics(a)
    print(f'A: layers={la} nodes={na:>5} embedded_np={maxna / 1e6:6.1f}MB '
          f'build_rss=+{rss_mb() - r0:5.0f}MB')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b_raw = open_dataarray(jp2, engine='rasterio').squeeze()
    print(f'B raw data type after unchunked open: {type(b_raw.data).__name__} '
          f'nbytes={b_raw.data.nbytes / 1e6:.0f}MB (rss +{rss_mb() - r0:.0f}MB)')
    b = b_raw.chunk({'y': 500, 'x': 500})
    lb, nb, _ = graph_metrics(b)
    print(f'B after .chunk(): layers={lb} nodes={nb:>5} '
          f'(rss +{rss_mb() - r0:.0f}MB total since before open)')

    t0 = time.perf_counter()
    va = a.compute()
    ta = time.perf_counter() - t0
    print(f'A full compute: {ta:5.1f}s  (rss now {rss_mb():.0f}MB)')

    t0 = time.perf_counter()
    vb = b.compute()
    tb = time.perf_counter() - t0
    print(f'B full compute: {tb:5.1f}s  (rss now {rss_mb():.0f}MB)')
    exact = bool((va == vb).all())
    print(f'bit-exact A==B: {exact}')
    del va, vb
    gc.collect()

    r0 = rss_mb()
    t0 = time.perf_counter()
    blk = a.isel(y=slice(2500, 3000), x=slice(2500, 3000)).compute()
    t_blk = time.perf_counter() - t0
    print(f'A single-block compute: shape={blk.shape} in {t_blk:.2f}s '
          f'rss_delta=+{rss_mb() - r0:.0f}MB (only that block read from disk)')

    ok = exact and na == 2 * (-(-ny // 500) * -(-nx // 500)) + 1
    print()
    print('VERDICT:', 'PASS — chunks at open are graph-cheap' if ok
          else 'CHECK — metrics deviate from expectation')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
