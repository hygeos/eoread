#!/usr/bin/env python3
"""One v1/v2 benchmark measurement, in a fresh process.

Launched by ``benchmark.py`` from inside the checkout of the branch under test::

    pixi run python3 run_one.py --branch {v1,v2} --sample-id ID --tier TIER \
        [--payload NAME] [--reader-kwargs '{"resolution": 20}']

so that ``import eoread`` resolves to *that* branch's code (the checkout dir,
given in the ``EOREAD_BENCH_REPO`` env var set by the driver, is prepended to
``sys.path``).  The product path comes from ``EOREAD_BENCH_SAMPLE``.

Exactly one result line is printed on stdout: ``@@BENCH@@ <json>``.  All other
output (reader verbose prints, warnings) is ignored by the driver.  The child
exits 0 on success and 1 on failure; either way the JSON line is printed, so a
crash never looks like "no output".

Tiers
-----
- ``probe`` : build + data-variable inventory (2D+ data vars: bytes/dims/shape/
              dtype, plus 1D coordinate descriptors with value hashes), no compute
- ``T0``    : build only (lazy; no pixel compute)
- ``T1``    : build + compute of the central S×S slice of the payload
              (S = ``EOREAD_BENCH_SLICE``, default 500 px)
- ``T2``    : build + full compute of the payload

Metrics: wall time of the build and of the whole measurement, plus the child
process peak RSS (``ru_maxrss`` high-water mark, MB — it includes the
interpreter + import baseline, identical on both branches).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

MARKER = '@@BENCH@@'

#: Sample id -> (module, callable) per branch.  v1 (master 5122d8b, eoread
#: 0.5.0): flat ``eoread/<sensor>.py`` modules with top-level functions.
#: v2 (dev): the ``eoread/readers/`` package; ``Class(path)`` returns the lazy
#: dataset through the Reader metaclass.  Every callable is invoked with the
#: product path only — pure defaults on both sides (v1 signature defaults,
#: v2 class ``chunks_default``).
READERS = {
    'MSI_L1C':           {'v1': ('eoread.msi', 'Level1_MSI'),
                          'v2': ('eoread.readers.msi', 'Level1C_MSI')},
    'MSI_L2A':           {'v1': ('eoread.msi', 'Level2_MSI'),
                          'v2': ('eoread.readers.msi', 'Level2A_MSI')},
    'OLCI_L1_EFR':       {'v1': ('eoread.olci', 'Level1_OLCI'),
                          'v2': ('eoread.readers.olci', 'Level1_OLCI')},
    'OLCI_L1_ERR':       {'v1': ('eoread.olci', 'Level1_OLCI'),
                          'v2': ('eoread.readers.olci', 'Level1_OLCI')},
    'OLCI_L2_WFR':       {'v1': ('eoread.olci', 'Level2_OLCI'),
                          'v2': ('eoread.readers.olci', 'Level2_OLCI')},
    'OLCI_L2_WRR':       {'v1': ('eoread.olci', 'Level2_OLCI'),
                          'v2': ('eoread.readers.olci', 'Level2_OLCI')},
    'MODIS_L1_AQUA':     {'v1': ('eoread.modis', 'Level1_MODIS'),
                          'v2': ('eoread.readers.modis', 'Level1B_MODIS')},
    'OLI_L1TP_L8':       {'v1': ('eoread.oli', 'Level1_OLI'),
                          'v2': ('eoread.readers.oli', 'Level1_OLI')},
    'OLI_L1TP_L9':       {'v1': ('eoread.oli', 'Level1_OLI'),
                          'v2': ('eoread.readers.oli', 'Level1_OLI')},
    'ECOSTRESS_L1C':     {'v1': ('eoread.ecostress', 'Level1C_ECOSTRESS'),
                          'v2': ('eoread.readers.ecostress', 'Level1C_ECOSTRESS')},
    # v1 has no L1B reader — the driver marks this sample not comparable.
    'ECOSTRESS_L1B':     {'v1': None,
                          'v2': ('eoread.readers.ecostress', 'Level1B_ECOSTRESS')},
    'ECOSTRESS_L2_V001': {'v1': ('eoread.ecostress', 'Level2_ECOSTRESS'),
                          'v2': ('eoread.readers.ecostress', 'Level2_ECOSTRESS')},
    'ECOSTRESS_L2_V002': {'v1': ('eoread.ecostress', 'Level2_ECOSTRESS'),
                          'v2': ('eoread.readers.ecostress', 'Level2_ECOSTRESS')},
    'MERIS_L1':          {'v1': ('eoread.meris', 'Level1_MERIS'),
                          'v2': ('eoread.readers.meris', 'Level1_MERIS')},
    'VENUS_L1C':         {'v1': ('eoread.venus', 'Level1_VENUS'),
                          'v2': ('eoread.readers.venus', 'Level1C_VENUS')},
    'VENUS_L2A':         {'v1': ('eoread.venus', 'Level2_VENUS'),
                          'v2': ('eoread.readers.venus', 'Level2A_VENUS')},
    'PACE_L1B':          {'v1': ('eoread.pace', 'Level1B_PACE_OCI'),
                          'v2': ('eoread.readers.pace', 'Level1B_PACE_OCI')},
    'SGLI_L1B':          {'v1': ('eoread.sgli', 'Level1_SGLI'),
                          'v2': ('eoread.readers.sgli', 'Level1B_SGLI')},
    'HICO_L1B':          {'v1': ('eoread.hico', 'Level1B_HICO'),
                          'v2': ('eoread.readers.hico', 'Level1B_HICO')},
    'PRISMA_L1':         {'v1': ('eoread.prisma', 'Level1_PRISMA'),
                          'v2': ('eoread.readers.prisma', 'Level1_PRISMA')},
    'GEOSAT_DE1':        {'v1': ('eoread.geosat', 'Level1_GEOSAT'),
                          'v2': ('eoread.readers.geosat', 'Level1_GEOSAT')},
    'GEOSAT_DE2':        {'v1': ('eoread.geosat', 'Level1_GEOSAT'),
                          'v2': ('eoread.readers.geosat', 'Level1_GEOSAT')},
    'PLEIADES_MS':       {'v1': ('eoread.pleiades', 'Level1_Pleiades'),
                          'v2': ('eoread.readers.pleiades', 'Level1_PLEIADES')},
    'PNEO_MS':           {'v1': ('eoread.pleiades_neo', 'Level1_PNEO'),
                          'v2': ('eoread.readers.pleiades_neo', 'Level1_PNEO')},
    'PLANETSCOPE_MS':    {'v1': ('eoread.planetscope', 'Level1_Planetscope'),
                          'v2': ('eoread.readers.planetscope', 'Level1_PLANETSCOPE')},
    'GOESNG_L1':         {'v1': ('eoread.goesng', 'Level1B_GOESNG'),
                          'v2': ('eoread.readers.goesng', 'Level1B_GOESNG')},
    'HYPSO_L1':          {'v1': ('eoread.hypso', 'Level1B_HYPSO'),
                          'v2': ('eoread.readers.hypso', 'Level1B_HYPSO')},
}

LATLON_NAMES = {'lat', 'lon', 'latitude', 'longitude'}


def emit(result: dict) -> None:
    print(f'{MARKER} {json.dumps(result)}', flush=True)
    sys.exit(0 if result.get('ok') else 1)


def build_reader(branch: str, sample_id: str, path: str,
                 reader_kwargs: dict | None = None):
    entry = READERS.get(sample_id)
    if entry is None or entry[branch] is None:
        raise LookupError(f'no {branch} reader registered for sample {sample_id!r}')
    module_name, attr = entry[branch]
    # v1 DIMAP readers (geosat/pleiades/planetscope) call .exists()/.glob() on
    # the product arg and require a pathlib.Path; all other v1 readers accept
    # str|Path, so passing Path everywhere is safe. v2 takes the registry's str.
    arg = Path(path) if branch == 'v1' else path
    return getattr(importlib.import_module(module_name), attr)(arg,
                                                               **(reader_kwargs or {}))


def _coord_descriptor(coord) -> dict | None:
    """Compact, exact descriptor of a 1D coordinate.

    The driver uses it to assert that two branches pin the same grid without
    shipping the values in the JSONL: sha256 over canonical bytes (ints ->
    int64, floats -> float64 with NaN pinned to 0, datetimes -> int64 ns,
    text -> utf-8 joined by NUL).  Small coordinates carry their values too,
    for readable diagnostics.
    """
    a = np.asarray(coord)
    if a.ndim != 1 or a.size == 0:
        return None
    flat = a.ravel()
    kind = a.dtype.kind
    desc = {'dtype': str(a.dtype), 'shape': [int(a.size)]}
    if kind in 'iu':
        canon, first, last = flat.astype('int64'), int(flat[0]), int(flat[-1])
    elif kind == 'f':
        canon = np.where(np.isnan(flat.astype('float64')), 0.0,
                         flat.astype('float64'))
        first, last = float(flat[0]), float(flat[-1])
    elif kind == 'M':
        canon, first, last = flat.astype('int64'), str(flat[0]), str(flat[-1])
    elif kind in 'USO':
        blob = '\x00'.join(str(x) for x in flat).encode('utf-8', 'replace')
        desc['sha256'] = hashlib.sha256(blob).hexdigest()
        desc['first'], desc['last'] = str(flat[0]), str(flat[-1])
        if a.size <= 256:
            desc['values'] = [str(x) for x in flat]
        return desc
    else:
        return None
    desc['sha256'] = hashlib.sha256(np.ascontiguousarray(canon).tobytes()).hexdigest()
    desc['first'], desc['last'] = first, last
    # Information-free index coordinate (0..n-1): the driver treats it as
    # equivalent to a bare dim with no coordinate at all.
    desc['arange'] = bool(np.array_equal(canon, np.arange(a.size)))
    if a.size <= 256:
        desc['values'] = [float(x) for x in canon] if kind == 'f' \
            else [int(x) for x in canon]
    return desc


def inventory(ds) -> dict:
    """Data-variable inventory + 1D coordinate descriptors. No compute.

    Returns ``{'vars': {name: {bytes, dims, shape, dtype}},
    'coords': {dim: descriptor or None}}`` over the 2D+ data variables (lat/lon
    excluded).  The driver asserts on the pinned variable that both branches
    pin the same workload: identical dims (order-insensitive), identical per-dim
    sizes (the bands count may be larger on v2), and matching coordinate values
    where both sides carry substantive ones; dtype differences are allowed
    (readers decode native dtypes, no casting).
    """
    vars_ = {}
    for name, var in ds.data_vars.items():
        if var.ndim >= 2 and str(name) not in LATLON_NAMES:
            vars_[str(name)] = {'bytes': int(var.nbytes),
                                'dims': [str(d) for d in var.dims],
                                'shape': [int(s) for s in var.shape],
                                'dtype': str(var.dtype)}
    coords = {}
    for dim in ds.dims:
        dim = str(dim)
        c = ds.coords[dim] if dim in ds.coords else None
        coords[dim] = _coord_descriptor(c) if (c is not None and c.dims == (dim,)) \
            else None
    return {'vars': vars_, 'coords': coords}


def main() -> None:
    ap = argparse.ArgumentParser(description='one v1/v2 benchmark measurement')
    ap.add_argument('--branch', required=True, choices=['v1', 'v2'])
    ap.add_argument('--sample-id', required=True)
    ap.add_argument('--tier', required=True, choices=['probe', 'T0', 'T1', 'T2'])
    ap.add_argument('--payload', default=None,
                    help='variable to pin (default: largest 2D+ data var)')
    ap.add_argument('--reader-kwargs', default=None,
                    help='JSON dict of extra reader arguments '
                         '(e.g. \'{"resolution": 20}\'), applied identically on '
                         'both branches by the driver')
    args = ap.parse_args()
    try:
        reader_kwargs = json.loads(args.reader_kwargs) if args.reader_kwargs else {}
        assert isinstance(reader_kwargs, dict)
    except (json.JSONDecodeError, AssertionError) as e:
        emit({'ok': False, 'error': f'bad --reader-kwargs JSON: {e}'})

    repo = os.environ.get('EOREAD_BENCH_REPO')
    sample_path = os.environ.get('EOREAD_BENCH_SAMPLE')
    if not repo or not sample_path:
        emit({'ok': False, 'error': 'EOREAD_BENCH_REPO / EOREAD_BENCH_SAMPLE not set'})

    sys.path.insert(0, repo)  # this branch's eoread package wins
    stubs = os.environ.get('EOREAD_BENCH_STUBS')
    if args.branch == 'v1' and stubs:
        sys.path.insert(0, stubs)  # jrc_rayleigh_processor import stand-in

    result = {'branch': args.branch, 'sample': args.sample_id, 'tier': args.tier,
              'payload': args.payload, 'ok': False}
    try:
        t_start = time.perf_counter()
        ds = build_reader(args.branch, args.sample_id, sample_path, reader_kwargs)
        wall_build = time.perf_counter() - t_start

        if args.tier == 'probe':
            result['inventory'] = inventory(ds)
        else:
            inv = inventory(ds)
            if args.payload is None:
                if not inv['vars']:
                    raise ValueError('no 2D+ data variable to pick a payload from')
                payload = max(inv['vars'], key=lambda n: inv['vars'][n]['bytes'])
            else:
                payload = args.payload
            result['payload'] = payload
            if payload not in ds.data_vars or ds[payload].ndim < 2:
                raise ValueError(f'pinned payload {payload!r} is missing or not 2D+')

            if args.tier == 'T1':
                w = int(os.environ.get('EOREAD_BENCH_SLICE', '500'))
                arr = ds[payload]
                # Central S×S over the two spatial dims, by name when they carry
                # the conventional 'y'/'x' labels (the bands dim may sit anywhere
                # in the order — HYPSO v1 ends with it), else the last two dims.
                if {'y', 'x'} <= set(arr.dims):
                    dy, dx = 'y', 'x'
                else:
                    dy, dx = arr.dims[-2], arr.dims[-1]
                ny, nx = arr.sizes[dy], arr.sizes[dx]
                w = min(w, ny, nx)
                cy, cx = ny // 2 - w // 2, nx // 2 - w // 2
                arr.isel({dy: slice(cy, cy + w), dx: slice(cx, cx + w)}).compute()
            elif args.tier == 'T2':
                ds[payload].compute()

        wall_total = time.perf_counter() - t_start
        result['ok'] = True
        result['wall_build_s'] = round(wall_build, 3)
        result['wall_total_s'] = round(wall_total, 3)
        result['peak_rss_mb'] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception as e:  # noqa: BLE001 — one JSON line even on failure
        result['error'] = f'{type(e).__name__}: {e}'
    emit(result)


if __name__ == '__main__':
    main()
