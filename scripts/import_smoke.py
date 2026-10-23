#!/usr/bin/env python3
"""
No-data gate, step 1: import smoke-check of every eoread module.

Imports every module of the package (top-level helpers, the v2 readers and
transforms, the legacy readers) without touching any product data. Any import
failure fails the gate — a commit can never leave the package unimportable in
a supported environment.

Modules are enumerated by walking the source tree for .py files. The previous
pkgutil.iter_modules walk silently skipped namespace subpackages (directories
without __init__.py), which dropped every reader module from this check while
eoread was still a namespace package (Phase-6 finding; the package is now a
regular one, but the explicit walk keeps the coverage obvious).

Usage: python3 scripts/import_smoke.py
"""

import importlib
import pathlib
import sys


def package_root() -> pathlib.Path:
    import eoread
    return pathlib.Path(next(iter(eoread.__path__)))


def collect_modules() -> list[str]:
    """Every .py file under the package root, as a dotted module name."""
    root = package_root()
    modules = set()
    for p in root.rglob('*.py'):
        parts = p.relative_to(root).parts
        if '__pycache__' in parts:
            continue
        dotted = '.'.join([*parts[:-1], parts[-1][:-3]])
        modules.add(f'eoread.{dotted}')
    return sorted(modules)


def main() -> int:
    modules = collect_modules()
    failed = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as e:  # the gate must report, not crash
            failed.append((name, f'{type(e).__name__}: {e}'))

    if failed:
        print(f'IMPORT SMOKE FAILED — {len(failed)} module(s) cannot be imported:')
        for name, err in failed:
            print(f'  {name}: {err}')
        return 1

    print(f'OK — all {len(modules)} eoread modules import cleanly')
    return 0


if __name__ == '__main__':
    sys.exit(main())
