#!/usr/bin/env python3
"""Stage the PRISMA L1 sample next to a synthesized Level2C companion.

The shared archive holds no Level2C for the bench granule, but both readers
require one for the per-pixel geometric angles (sza/vza/raa). And v1's reader
resolves the product path, so a symlinked granule would escape the staging
directory and lose the companion — the L1 must be a *real copy* here.

This stages that pair under ``<stagedir>``:

- a real copy of the L1 (reused when already present);
- a synthesized L2C with constant geometric grids — sza taken from the L1
  global ``Sun_zenith_angle``, vza/raa fixed — mirroring the fixture in
  ``tests/test_prisma.py`` (idempotent: an existing or real L2C wins).

Invoked by ``benchmark.py`` inside the v2 pixi env (needs h5py/numpy)::

    python3 stage_prisma.py <l1.he5> <stagedir>

Prints the staged L1 path on the last stdout line; progress goes to stderr.
"""

import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def synthesize_l2c(l1: Path, l2c: Path) -> None:
    """Write a constant-grid Level2C companion for the staged L1 granule."""
    with h5py.File(l1, 'r') as f:
        cube = f['HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube']
        ny, nx = cube.shape[0], cube.shape[2]
        raw_sza = f.attrs['Sun_zenith_angle'] if 'Sun_zenith_angle' in f.attrs else 30.0
        sza = float(np.asarray(raw_sza).item())
    with h5py.File(l2c, 'w') as f:
        geom = f.create_group('HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields')
        for name, value in [('Solar_Zenith_Angle', sza),
                            ('Observing_Angle', 15.0),
                            ('Rel_Azimuth_Angle', 45.0)]:
            geom.create_dataset(name, data=np.full((ny, nx), value, dtype='float32'))


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f'usage: {sys.argv[0]} <l1.he5> <stagedir>')
    src = Path(sys.argv[1])
    stagedir = Path(sys.argv[2])
    stagedir.mkdir(parents=True, exist_ok=True)

    l1 = stagedir / src.name
    if not l1.exists():
        # Real copy (not a symlink): v1 resolves the product path, so a
        # symlink would escape the staging dir and lose the companion.
        print(f'copying {src} -> {l1}', file=sys.stderr)
        shutil.copy2(src, l1)
        print(f'staged L1: {l1.stat().st_size / 2**30:.2f} GB', file=sys.stderr)

    l2c = l1.with_name(l1.name.replace('PRS_L1_STD_OFFL_', 'PRS_L2C_STD_', 1))
    if not l2c.exists():
        print(f'synthesizing {l2c}', file=sys.stderr)
        synthesize_l2c(l1, l2c)

    print(l1)


if __name__ == '__main__':
    main()
