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
import re
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


def test_class_names_encode_level():
    """Class names must be exactly `Level{level}_{SLUG}`.

    The level part of the name mirrors the `level` class attribute with its
    leading 'L' stripped (level 'L1C' -> 'Level1C_...'), so a bare 'Level1_'
    name can only exist for a reader whose product level really is 'L1'. The
    sensor slug is uppercase [A-Z0-9_]+; it need not equal the `sensor`
    display attribute (e.g. Level1B_GOESNG carries sensor 'ABI').
    """
    for cls in discover_readers():
        level = cls.level or ''
        assert level.startswith('L'), f'{cls.__name__}: level {level!r} must start with "L"'
        prefix = f'Level{level[1:]}_'
        assert cls.__name__.startswith(prefix), (
            f'{cls.__name__}: name must start with {prefix!r} '
            f'(level attribute is {level!r})')
        slug = cls.__name__[len(prefix):]
        assert re.fullmatch(r'[A-Z][A-Z0-9_]*', slug), (
            f'{cls.__name__}: sensor slug {slug!r} must match [A-Z][A-Z0-9_]*')
