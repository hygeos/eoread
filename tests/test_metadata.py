#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the metadata template filtering."""

import pytest

from eoread.tools import filter_metadata


def test_filter_metadata_keeps_siblings_under_same_parent():
    metadata = {'a': 0, 'b': {'c': 1, 'd': 2, 'e': {'f': 3, 'g': 4}}}

    result = filter_metadata(metadata, [
        ['a'],
        ['b', 'c'],
        ['b', 'd'],
        ['b', 'e', 'f'],
        ['b', 'e', 'g'],
    ])

    assert result == metadata


def test_filter_metadata_single_leaf():
    metadata = {'a': 0, 'b': {'c': 1, 'd': 2}}
    assert filter_metadata(metadata, [['b', 'c']]) == {'b': {'c': 1}}


def test_filter_metadata_empty_template():
    assert filter_metadata({'a': 0}, []) == {}


def test_filter_metadata_missing_leaf_is_loud():
    with pytest.raises(ValueError, match='Leaf d not found'):
        filter_metadata({'a': {'b': 1}}, [['a', 'd']])


def test_filter_metadata_conflicting_paths_are_loud():
    with pytest.raises(ValueError, match='conflicting template paths'):
        filter_metadata({'a': 1}, [['a'], ['a', 'b']])
