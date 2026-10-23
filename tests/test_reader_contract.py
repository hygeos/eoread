#!/usr/bin/env python3
"""
Static contract tests for `eoread.reader.Reader`.

Every `read()` implementation must end with ``return self.finalize(ds)`` so
that metadata, flag definitions, `schema_version` and `validate()` are applied
exactly once, in the reader. `_ReaderMeta` enforces the contract at runtime by
asserting the returned dataset was finalized; these tests catch a violation by
inspection, without needing a sample product.
"""

import ast
import inspect
import textwrap

import pytest
import xarray as xr

from eoread.reader import Reader, discover_readers


def _last_statement(func):
    source = textwrap.dedent(inspect.getsource(func))
    return ast.parse(source).body[0].body[-1]


def _is_finalize_return(node) -> bool:
    return (
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == 'finalize'
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == 'self'
    )


def test_every_read_ends_with_finalize():
    readers = discover_readers()
    assert readers, 'no registered readers discovered'
    for cls in readers:
        last = _last_statement(cls.read)
        assert _is_finalize_return(last), (
            f'{cls.__name__}.read() must end with return self.finalize(ds); '
            f'got: {ast.unparse(last)}')


def test_meta_rejects_unfinalized_read():
    class Unfinalized(Reader):
        sensor = 'TEST'
        level = 'L0'

        def read(self, path):
            return xr.Dataset()

    with pytest.raises(AssertionError, match='must end with self.finalize'):
        Unfinalized('does-not-exist')
