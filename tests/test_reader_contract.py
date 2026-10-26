#!/usr/bin/env python3
"""
Static contract tests for `eoread.reader.Reader`.

Every `read()` implementation must end with ``return self.finalize(ds)`` so
that metadata, flag definitions, `schema_version` and `validate()` are applied
exactly once, in the reader. The contract is not enforced at runtime (there is
no check in `_ReaderMeta.__call__`); these tests catch a violation by source
inspection, without needing a sample product.
"""

import ast
import inspect
import textwrap

from eoread.reader import discover_readers


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
