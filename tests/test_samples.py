#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the out-of-reader sample-product registry."""

import pytest

import eoread.samples as samples_module
from eoread.reader import discover_readers
from eoread.readers.msi import Level1C_MSI
from eoread.readers.oli import Level1_OLI
from eoread.samples import Sample, available


def test_every_reader_declares_registered_sample_products():
    registry = available()
    for cls in discover_readers():
        assert cls.sample_products, f'{cls.__name__} does not declare sample_products'
        for name in cls.sample_products:
            assert name in registry, (
                f'{cls.__name__} references unknown sample product {name!r}')


def test_sample_resolves_existing_path(tmp_path):
    sample = tmp_path / 'sample.SAFE'
    sample.mkdir()

    assert Sample(sample).resolve() == sample


def test_resolve_missing_path_is_loud(tmp_path):
    missing = tmp_path / 'missing.SAFE'
    with pytest.raises(FileNotFoundError, match='does not exist'):
        Sample(missing, 'test product').resolve()


def test_reader_get_sample_explicit_product(monkeypatch, tmp_path):
    sample = tmp_path / 'l9'
    sample.mkdir()
    monkeypatch.setitem(samples_module._samples, 'OLI_L1TP_L9', Sample(sample))

    assert Level1_OLI.get_sample('OLI_L1TP_L9') == sample


def test_reader_get_sample_falls_back_to_next_product(monkeypatch, tmp_path):
    sample = tmp_path / 'l9'
    sample.mkdir()
    monkeypatch.setitem(
        samples_module._samples, 'OLI_L1TP_L8', Sample(tmp_path / 'missing'))
    monkeypatch.setitem(samples_module._samples, 'OLI_L1TP_L9', Sample(sample))

    assert Level1_OLI.get_sample() == sample


def test_reader_get_sample_reports_all_candidates(monkeypatch, tmp_path):
    monkeypatch.setitem(
        samples_module._samples, 'OLI_L1TP_L8', Sample(tmp_path / 'missing-l8'))
    monkeypatch.setitem(
        samples_module._samples, 'OLI_L1TP_L9', Sample(tmp_path / 'missing-l9'))

    with pytest.raises(FileNotFoundError) as exc:
        Level1_OLI.get_sample()
    message = str(exc.value)
    assert 'missing-l8' in message
    assert 'missing-l9' in message


def test_unknown_sample_id_is_loud():
    with pytest.raises(KeyError, match='not part of declared sample products'):
        Level1C_MSI.get_sample('NOPE')


def test_sample_is_immutable(tmp_path):
    sample = Sample(tmp_path)
    with pytest.raises(Exception):
        sample.path = tmp_path / 'other'
