#!/usr/bin/env python3
"""
No-data gate, step 1: import smoke-check of every eoread module.

Imports every module of the package (top-level helpers, the v2 skeleton, the
legacy readers and their subpackages) without touching any product data. Any
import failure fails the gate — a commit can never leave the package
unimportable in a supported environment.

Usage: python3 scripts/import_smoke.py
"""

import importlib
import pkgutil
import sys


def collect_modules() -> list[str]:
    """List every submodule of eoread (one level below each subpackage)."""
    modules = []
    top = importlib.import_module('eoread')
    for info in pkgutil.iter_modules(top.__path__):
        full = f'eoread.{info.name}'
        modules.append(full)
        if info.ispkg:
            sub = importlib.import_module(full)
            modules.extend(f'{full}.{s.name}' for s in pkgutil.iter_modules(sub.__path__))
    return modules


def main() -> int:
    failed = []
    for name in sorted(collect_modules()):
        try:
            importlib.import_module(name)
        except Exception as e:  # the gate must report, not crash
            failed.append((name, f'{type(e).__name__}: {e}'))

    if failed:
        print(f'IMPORT SMOKE FAILED — {len(failed)} module(s) cannot be imported:')
        for name, err in failed:
            print(f'  {name}: {err}')
        return 1

    count = len(collect_modules())
    print(f'OK — all {count} eoread modules import cleanly')
    return 0


if __name__ == '__main__':
    sys.exit(main())
