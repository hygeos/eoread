#!/usr/bin/env python3
"""Driver for the eoread v1 (master, 0.5.0) vs v2 (dev) time/RAM benchmark.

Host-side script — stdlib only.  For each sample product in the v2 branch's
``eoread.samples`` registry it:

1. probes both branches (one fresh subprocess each): build + data-var
   inventory (bytes/dims/shape/dtype + 1D coordinate descriptors), no compute;
2. pins the payload per branch: a declared preference when the row has one
   (e.g. MSI `Rtoa`/`Ltoa`), else the largest *common* variable — asserting in
   both cases that both branches pin the same workload: identical dims
   (order-insensitive) and per-dim sizes, where the bands count may be larger
   on v2 (superset of bands); coordinate labels are compared only when both
   sides carry substantive ones.  dtype differences are allowed (native
   dtypes are not cast).
   A failing assertion marks the row not comparable, with its reason;
3. measures T0/T1/T2 on both branches for R rounds — one fresh subprocess per
   measurement, launched as ``pixi run python3 run_one.py ...`` inside each
   branch's own checkout so each side runs in its own pixi env;
4. writes ``results/<UTC stamp>/results.jsonl`` + ``report.md`` (medians and
   v2/v1 ratios).

Usage::

    python3 scripts/benchmark/benchmark.py full
    python3 scripts/benchmark/benchmark.py full --rounds 1 --samples PACE_L1B
    python3 scripts/benchmark/benchmark.py full --dry-run

MSI is measured per resolution (rows MSI_L1C_60m/20m/10m, same for L2A): the
v1 reader defaults to the 60 m grid, the v2 one to the 10 m grid, so comparing
defaults would pit a ~36x smaller workload against each other.  ``--samples``
accepts a base id (expands to all its variant rows) or an exact row id.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # v2 checkout containing this script
RUN_ONE = HERE / 'run_one.py'
STUBS = HERE / '_stubs'
#: v1 checkout convention: sibling of the v2 repo, named `eoread-v1`
#: (from the parent dir: ./eoread-v1 vs ./eoread).
V1_DEFAULT_PATH = REPO_ROOT.parent / 'eoread-v1'

MARKER = '@@BENCH@@'
TIERS = ('T0', 'T1', 'T2')
TIMEOUTS = {'probe': 600, 'T0': 900, 'T1': 900, 'T2': 3600}

#: Sample id -> staging script for products that need auxiliary files next to
#: the granule before either branch can read it. The script is run in the v2
#: pixi env (it may use the project's dependencies), takes
#: ``<product path> <staging dir>``, must be idempotent, and prints the staged
#: product path on its last stdout line.
STAGED_SAMPLES = {
    # PRISMA L1 needs a sibling Level2C companion for the per-pixel angles,
    # which the archive does not hold; v1's reader resolves the product path,
    # so the granule must be a real copy in the staging dir (a symlink would
    # escape it and lose the companion). See stage_prisma.py.
    'PRISMA_L1': HERE / 'stage_prisma.py',
}

#: Sample id -> measurement variants for products whose two branches need an
#: explicit reader argument to do the same job.  Row id = sample id + suffix;
#: each variant is (suffix, reader kwargs applied identically to both branches,
#: preferred payload names per branch or None).
#: MSI: v1 defaults to the 60 m grid, v2 to the 10 m one — every resolution is
#: therefore pinned explicitly on both sides.  The L1C payload names differ by
#: design (legacy `Rtoa`, v2 `Ltoa`); if the preferred pair fails the
#: same-workload assertion the row is reported not comparable with the reason
#: (no fallback to a smaller variable).
VARIANTS = {
    'MSI_L1C': [('_60m', {'resolution': 60}, {'v1': 'Rtoa', 'v2': 'Ltoa'}),
                ('_20m', {'resolution': 20}, {'v1': 'Rtoa', 'v2': 'Ltoa'}),
                ('_10m', {'resolution': 10}, {'v1': 'Rtoa', 'v2': 'Ltoa'})],
    'MSI_L2A': [('_60m', {'resolution': 60}, {'v1': 'Rtoa', 'v2': 'Rtoa'}),
                ('_20m', {'resolution': 20}, {'v1': 'Rtoa', 'v2': 'Rtoa'}),
                ('_10m', {'resolution': 10}, {'v1': 'Rtoa', 'v2': 'Rtoa'})],
}


def variants_of(sid: str) -> list[tuple[str, dict, dict | None]]:
    return VARIANTS.get(sid, [('', {}, None)])

# Paths are resolved against the v2 checkout (this snippet runs with cwd =
# v2_dir): some samples live in the v2 repo itself (e.g. the bundled OLCI EFR
# granule under data/sample_products/), which does not exist in the v1
# checkout where run_one.py resolves them from its own cwd.
DISCOVER_SNIPPET = (
    'import json; from pathlib import Path; from eoread.samples import available; '
    'print(json.dumps({k: str(Path(s.path).expanduser().resolve()) '
    'for k, s in available().items()}))'
)


def die(msg: str) -> None:
    print(f'error: {msg}', file=sys.stderr)
    sys.exit(2)


def git_sha(directory: Path) -> str:
    try:
        out = subprocess.run(['git', '-C', str(directory), 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def discover_samples(v2_dir: Path) -> dict[str, str]:
    """Sample id -> product path, from the v2 branch's ``eoread.samples``."""
    p = subprocess.run(['pixi', 'run', 'python3', '-c', DISCOVER_SNIPPET],
                       cwd=v2_dir, capture_output=True, text=True,
                       timeout=TIMEOUTS['probe'])
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if line.startswith('{'):
            return dict(sorted(json.loads(line).items()))
    die(f'could not discover samples from {v2_dir}:\n{p.stderr[-800:]}')


def stage_samples(v2_dir: Path, samples: dict[str, str]) -> None:
    """Re-point staged samples at their staged product path (in place).

    Runs after --dry-run returns, so a dry run never copies anything.
    """
    for sid, script in STAGED_SAMPLES.items():
        if sid not in samples:
            continue
        stagedir = HERE / '_staged' / sid
        p = subprocess.run(['pixi', 'run', 'python3', str(script), samples[sid],
                            str(stagedir)],
                           cwd=v2_dir, capture_output=True, text=True,
                           timeout=TIMEOUTS['probe'])
        if p.returncode != 0:
            die(f'staging {sid} failed:\n{p.stderr[-800:]}')
        lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
        samples[sid] = lines[-1]
        print(f'[stage] {sid}: {samples[sid]}')


def run_one(branch_dir: Path, branch: str, sample_id: str, path: str,
            tier: str, payload: str | None, slice_w: int,
            reader_kwargs: dict | None = None) -> dict:
    """Launch one measurement subprocess and parse its JSON result line."""
    cmd = ['pixi', 'run', 'python3', str(RUN_ONE),
           '--branch', branch, '--sample-id', sample_id, '--tier', tier]
    if payload is not None:
        cmd += ['--payload', payload]
    if reader_kwargs:
        cmd += ['--reader-kwargs', json.dumps(reader_kwargs)]
    env = {**os.environ,
           'EOREAD_BENCH_REPO': str(branch_dir),
           'EOREAD_BENCH_SAMPLE': path,
           'EOREAD_BENCH_STUBS': str(STUBS),
           'EOREAD_BENCH_SLICE': str(slice_w)}
    try:
        p = subprocess.run(cmd, cwd=branch_dir, env=env, capture_output=True,
                           text=True, timeout=TIMEOUTS[tier])
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': f'timeout after {TIMEOUTS[tier]} s'}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(MARKER + ' '):
            try:
                res = json.loads(line[len(MARKER) + 1:])
            except json.JSONDecodeError as e:
                return {'ok': False, 'error': f'bad result JSON: {e}'}
            if not res.get('ok'):
                res.setdefault('stderr_tail', p.stderr[-800:])
            return res
    return {'ok': False, 'error': f'no result line (rc={p.returncode})',
            'stderr_tail': p.stderr[-800:]}


def median(xs):
    return statistics.median(xs) if xs else None


def fmt(v, spec='.1f', fail='FAIL'):
    return fail if v is None else format(v, spec)


def payload_mismatch(v1v: dict, v2v: dict, c1: dict, c2: dict) -> str | None:
    """Why the two pinned variables do not pin the same workload, or None.

    Dims must match as a set — order is irrelevant.  Every dim except
    ``bands`` must carry an identical size on both sides; the band count may
    differ only in v2's favour (v2 reads the same or more bands: its workload
    is a superset of v1's, e.g. where legacy dropped red-edge bands).
    Coordinate values are grid labels, not data: they are never compared for
    ``bands``, and for other dims a coordinate present on only one side (the
    other side bare or trivial 0..n-1) is a labeling difference — the dim
    exists at equal size on both.  When both sides carry substantive coords,
    the values must still match (a same-size but shifted crop would be a
    different workload).  dtype differences are allowed (readers decode
    native dtypes, no casting).
    """
    d1 = dict(zip(v1v['dims'], v1v['shape']))
    d2 = dict(zip(v2v['dims'], v2v['shape']))
    if set(d1) != set(d2):
        return f"dims {v1v['dims']} != {v2v['dims']}"
    for d, s1 in d1.items():
        s2 = d2[d]
        if d == 'bands':
            if s2 < s1:
                return f'bands count v1 {s1} > v2 {s2}'
        elif s1 != s2:
            return f"dim {d!r} size v1 {s1} != v2 {s2}"
    for d in d1:
        if d == 'bands':
            continue  # band labels are never compared
        a, b = c1.get(d), c2.get(d)
        # A trivial 0..n-1 index coordinate carries no grid information —
        # treat it as equivalent to the dim having no coordinate at all.
        if a is not None and a.get('arange'):
            a = None
        if b is not None and b.get('arange'):
            b = None
        if a is None or b is None:
            continue  # labeling difference only — see docstring
        if a['sha256'] != b['sha256']:
            return (f'coordinate {d!r} values differ '
                    f'(v1 {a["first"]}..{a["last"]} / v2 {b["first"]}..{b["last"]})')
    return None


def choose_payloads(inv1: dict, inv2: dict, pref: dict | None = None):
    """Pick the payload per branch: (payloads, reason).

    Strict rule — only the primary candidate is eligible: a declared
    preference when the row has one (names may differ per branch, MSI L1C:
    legacy `Rtoa` vs v2 `Ltoa`), else the largest common variable
    (deterministic tie-break by name).  If it fails the same-workload
    assertion the row is not comparable — no silent fallback to a smaller
    variable.  payloads = {'v1': name, 'v2': name} or None; reason is the
    failure message (for the report) or None.
    """
    v1vars, v2vars = inv1['vars'], inv2['vars']
    c1, c2 = inv1['coords'], inv2['coords']
    if pref:
        n1, n2 = pref['v1'], pref['v2']
        if n1 not in v1vars or n2 not in v2vars:
            missing = ([n1] if n1 not in v1vars else []) + \
                      ([n2] if n2 not in v2vars else [])
            return None, f'preferred payload missing: {", ".join(missing)}'
        why = payload_mismatch(v1vars[n1], v2vars[n2], c1, c2)
        if why is not None:
            return None, f'preferred {n1}/{n2}: {why}'
        return {'v1': n1, 'v2': n2}, None
    common = set(v1vars) & set(v2vars)
    if not common:
        v1names = ', '.join(sorted(v1vars)) or '(none)'
        v2names = ', '.join(sorted(v2vars)) or '(none)'
        return None, f'no common 2D+ data variable (v1: {v1names} · v2: {v2names})'
    name = sorted(common, key=lambda n: (-(v1vars[n]['bytes'] + v2vars[n]['bytes']), n))[0]
    why = payload_mismatch(v1vars[name], v2vars[name], c1, c2)
    if why is not None:
        return None, f'{name}: {why}'
    return {'v1': name, 'v2': name}, None


def write_report(outdir: Path, meta: dict, payloads: dict, notes: dict,
                 results: list[dict], incomplete: bool = False) -> None:
    title = '# eoread v1 vs v2 — time & RAM benchmark'
    if incomplete:
        title += '  *(partial — interrupted)*'
    lines = [title, '',
             f'- generated: {meta["stamp"]}',
             f'- v1: commit `{meta["v1_sha"]}` @ `{meta["v1_path"]}`',
             f'- v2: commit `{meta["v2_sha"]}` @ `{meta["v2_path"]}`',
             f'- rounds: {meta["rounds"]} · T1 slice: {meta["slice"]}×{meta["slice"]} px · '
             'ratio = median(v2 total) / median(v1 total) (< 1 → v2 faster)',
             '- payload assertion: identical dims (order-insensitive) and per-dim sizes on both '
             'branches; the bands count may be larger on v2 (superset of bands); coordinate labels '
             'compared only when both sides carry substantive ones (trivial 0..n-1 index coords '
             'count as absent); dtype differences are allowed (native dtypes, no casting)',
             '', '## Comparable rows', '']

    idx: dict[tuple, list[dict]] = {}
    for r in results:
        if r.get('ok'):
            idx.setdefault((r['row'], r['branch'], r['tier']), []).append(r)

    empty_inv = {'vars': {}, 'coords': {}}
    for rid in sorted(payloads):
        inv1, inv2 = meta['probe_inv'].get(rid, (empty_inv, empty_inv))
        pay = payloads[rid]
        n1, n2 = pay['v1'], pay['v2']
        v1e = inv1.get('vars', {}).get(n1)
        v2e = inv2.get('vars', {}).get(n2)
        head = f'### {rid} — payload ' + (f'`{n1}` / `{n2}`' if n1 != n2 else f'`{n1}`')
        if v1e and v2e:
            head += (f' ({v1e["bytes"] / 2**30:.2f} GB {v1e["dtype"]} / '
                     f'{v2e["bytes"] / 2**30:.2f} GB {v2e["dtype"]})')
        lines += [head, '',
                  '| tier | v1 build s | v1 total s | v2 build s | v2 total s '
                  '| ratio v2/v1 | v1 peak MB | v2 peak MB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for tier in TIERS:
            a = idx.get((rid, 'v1', tier), [])
            b = idx.get((rid, 'v2', tier), [])
            ab = median([r['wall_build_s'] for r in a])
            at = median([r['wall_total_s'] for r in a])
            bb = median([r['wall_build_s'] for r in b])
            bt = median([r['wall_total_s'] for r in b])
            ratio = (bt / at) if (at and bt) else None
            ra = median([r['peak_rss_mb'] for r in a])
            rb = median([r['peak_rss_mb'] for r in b])
            lines.append(f'| {tier} | {fmt(ab)} | {fmt(at)} | {fmt(bb)} | {fmt(bt)} '
                         f'| {fmt(ratio, ".2f", "—")} | {fmt(ra, ".0f")} | {fmt(rb, ".0f")} |')
        lines.append('')

    if notes:
        lines += ['## Not comparable', '', '| row | reason |', '|---|---|']
        for rid in sorted(notes):
            lines.append(f'| {rid} | {notes[rid]} |')
        lines.append('')

    (outdir / 'report.md').write_text('\n'.join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description='eoread v1 vs v2 time/RAM benchmark driver')
    ap.add_argument('mode', nargs='?', default='full', choices=['full'])
    ap.add_argument('--v1-path', default=str(V1_DEFAULT_PATH),
                    help='v1 (master 5122d8b) checkout, by convention the sibling '
                         f'directory of the v2 repo (default: {V1_DEFAULT_PATH})')
    ap.add_argument('--v2-path', default=str(REPO_ROOT),
                    help='v2 (dev) checkout — the repo containing this script (default: auto)')
    ap.add_argument('--rounds', type=int, default=5,
                    help='measurement rounds per row; single-round ratios are '
                         'coin-flips on some rows (default: 5)')
    ap.add_argument('--slice', type=int, default=500, dest='slice_w',
                    help='T1 central slice width in px (default: 500)')
    ap.add_argument('--samples', default=None,
                    help='comma-separated sample ids to restrict the run')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the measurement plan without launching anything')
    args = ap.parse_args()

    if shutil.which('pixi') is None:
        die('pixi not found on PATH')
    v1_dir = Path(args.v1_path).expanduser().resolve()
    v2_dir = Path(args.v2_path).expanduser().resolve()
    for d, label in ((v1_dir, 'v1'), (v2_dir, 'v2')):
        if not (d / 'pyproject.toml').exists():
            die(f'{label} checkout not found at {d} — see README.md setup')

    v1_sha, v2_sha = git_sha(v1_dir), git_sha(v2_dir)
    samples = discover_samples(v2_dir)
    # Expand per-variant measurement rows (MSI: one row per resolution).  Row
    # id = base sample id + suffix; the product registry stays keyed by the
    # base sample id.
    rows = {sid + suf: (sid, path, kw, pref)
            for sid, path in samples.items()
            for suf, kw, pref in variants_of(sid)}
    if args.samples:
        wanted = [s.strip() for s in args.samples.split(',') if s.strip()]
        unknown = [s for s in wanted if s not in rows and s not in samples]
        if unknown:
            die(f'unknown sample id(s): {", ".join(unknown)} '
                f'(known: {", ".join(sorted(set(samples) | set(rows)))})')
        rows = {r: v for r, v in sorted(rows.items()) if r in wanted or v[0] in wanted}

    n_meas = len(rows) * (2 + args.rounds * len(TIERS) * 2)
    print(f'rows: {len(rows)} measurement rows · v1 {v1_sha} @ {v1_dir}')
    print(f'          v2 {v2_sha} @ {v2_dir}')
    print(f'plan: 2 probes/row + {args.rounds} rounds × {len(TIERS)} tiers × 2 branches '
          f'= {n_meas} subprocesses (timeouts s: probe/T0/T1/T2 = {TIMEOUTS})')
    if args.dry_run:
        for rid in sorted(rows):
            _sid, path, kw, _pref = rows[rid]
            extra = f' reader_kwargs={kw}' if kw else ''
            print(f'  - {rid}: {path}{extra}')
        return

    # Stage only the base samples actually measured (re-points paths in place).
    sub = {sid: samples[sid] for sid, _p, _kw, _pref in rows.values()}
    stage_samples(v2_dir, sub)
    rows = {r: (sid, sub[sid], kw, pref) for r, (sid, _p, kw, pref) in rows.items()}

    outdir = HERE / 'results' / time.strftime('%Y%m%d-%H%M%S')
    outdir.mkdir(parents=True)
    jsonl = outdir / 'results.jsonl'

    def log(rec: dict) -> None:
        with jsonl.open('a') as f:
            f.write(json.dumps(rec) + '\n')

    # -- 1. probes + same-workload payload pinning ----------------------------
    payloads: dict[str, dict[str, str]] = {}
    notes: dict[str, str] = {}
    probe_inv: dict[str, tuple[dict, dict]] = {}
    for rid in sorted(rows):
        sid, path, kw, pref = rows[rid]
        p1 = run_one(v1_dir, 'v1', sid, path, 'probe', None, args.slice_w, kw)
        p2 = run_one(v2_dir, 'v2', sid, path, 'probe', None, args.slice_w, kw)
        for r in (p1, p2):
            r['round'] = 0
            r['row'] = rid
            log(r)
        nv1 = len(p1.get('inventory', {}).get('vars', {}))
        nv2 = len(p2.get('inventory', {}).get('vars', {}))
        print(f'[probe] {rid:<18} v1 {"ok " if p1.get("ok") else "FAIL"} ({nv1} vars) · '
              f'v2 {"ok " if p2.get("ok") else "FAIL"} ({nv2} vars)')
        if not p1.get('ok') or not p2.get('ok'):
            bad = next(r for r in (p1, p2) if not r.get('ok'))
            notes[rid] = f'probe failed on {bad["branch"]}: {bad.get("error", "?")}'
            continue
        probe_inv[rid] = (p1['inventory'], p2['inventory'])
        pays, reason = choose_payloads(p1['inventory'], p2['inventory'], pref)
        if pays is None:
            notes[rid] = reason or 'no common 2D+ data variable'
            continue
        payloads[rid] = pays
        print(f'          → payload v1 `{pays["v1"]}` / v2 `{pays["v2"]}`')

    if not payloads:
        die('no comparable row — see notes above')

    # -- 2. measurement rounds -------------------------------------------------
    meta = {'stamp': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
            'v1_sha': v1_sha, 'v2_sha': v2_sha,
            'v1_path': str(v1_dir), 'v2_path': str(v2_dir),
            'rounds': args.rounds, 'slice': args.slice_w, 'probe_inv': probe_inv}

    results: list[dict] = []
    try:
        for r in range(1, args.rounds + 1):
            for i, rid in enumerate(sorted(payloads)):
                sid, path, kw, _pref = rows[rid]
                # alternate branch order to counter machine drift
                order = ('v1', 'v2') if (r + i) % 2 == 0 else ('v2', 'v1')
                for tier in TIERS:
                    for b in order:
                        res = run_one(v1_dir if b == 'v1' else v2_dir, b, sid, path,
                                      tier, payloads[rid][b], args.slice_w, kw)
                        res['round'] = r
                        res['row'] = rid
                        results.append(res)
                        log(res)
                        if res.get('ok') and res.get('payload') != payloads[rid][b]:
                            print(f'  WARNING: {rid} {b} {tier} payload mismatch '
                                  f'{res["payload"]!r} != pinned {payloads[rid][b]!r}')
                        if res.get('ok'):
                            extra = (f"build={res['wall_build_s']:8.1f}s "
                                     f"total={res['wall_total_s']:8.1f}s "
                                     f"rss={res['peak_rss_mb']:8.0f}MB")
                        else:
                            extra = res.get('error', '?')[:70]
                        status = 'ok   ' if res.get('ok') else 'FAIL '
                        print(f'[r{r}/{args.rounds}] {rid:<18} {b} {tier}  {status} {extra}')
            write_report(outdir, meta, payloads, notes, results)
    except KeyboardInterrupt:
        print('\ninterrupted — writing partial report')
        write_report(outdir, meta, payloads, notes, results, incomplete=True)
        sys.exit(130)

    write_report(outdir, meta, payloads, notes, results)
    print(f'\ndone. report: {outdir / "report.md"}\nraw data: {jsonl}')


if __name__ == '__main__':
    main()
