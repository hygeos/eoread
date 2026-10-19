"""
eoread — modular satellite data input/output built on xarray (see README.md).

Layout (REFACTORING.md):
    readers/     v2 readers, one module per sensor, emitting the canonical schema
    transforms/  shared v2 transforms (geometry, radiometry, resampling, angles)
    legacy/      pre-v2 reader modules retained as reference during the migration

The package is a regular package by design: every ``__init__.py`` is
docstring-only and re-exports nothing. The absence of re-exports is deliberate
(upgrading from v1 is an explicit import update, never a silent alias) — see
REFACTORING.md §9.
"""
