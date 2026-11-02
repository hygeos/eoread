#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the out-of-reader sample-product registry."""

from pathlib import Path

import pytest

import eoread.samples as samples_module
from eoread.reader import discover_readers
from eoread.readers.msi import Level1C_MSI
from eoread.readers.oli import Level1_OLI
from eoread.samples import Download, Sample, available


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


# --------------------------------------------------------------------------
# Opt-in SAND download (Phase 4)
# --------------------------------------------------------------------------

def _missing_sample(provider='cdse', product_id='PID'):
    return Sample(Path('/nonexistent/sample'), 'test product',
                  download=Download(provider, product_id))


class _StubDownloader:
    """Stand-in for a SAND provider downloader (no network)."""

    def __init__(self, error=None, skip_create=False):
        self.error = error
        self.skip_create = skip_create
        self.calls = []

    def download_file(self, product_id, directory):
        self.calls.append((product_id, Path(directory)))
        if self.error is not None:
            raise self.error
        target = Path(directory) / product_id
        if not self.skip_create:
            target.mkdir(parents=True, exist_ok=True)
        return target


def test_download_local_path_wins(monkeypatch, tmp_path):
    # A present local path must never trigger a download.
    local = tmp_path / 'sample.SAFE'
    local.mkdir()

    def boom(sample):
        raise AssertionError('download must not be attempted')

    monkeypatch.setattr(samples_module, '_download_sample', boom)
    sample = Sample(local, 'x', download=Download('cdse', 'PID'))
    assert sample.resolve() == local


def test_download_registry_opt_in_list():
    # Only the explicitly registered products are downloadable.
    downloadable = {name for name, s in samples_module._samples.items()
                    if s.download is not None}
    assert downloadable == {'MSI_L1C', 'MSI_L2A', 'ECOSTRESS_L1C',
                            'VENUS_L1C', 'VENUS_L2A'}
    for name in downloadable:
        spec = samples_module._samples[name].download
        assert spec.provider in samples_module._PROVIDERS
        assert spec.product_id


def test_download_unknown_provider_is_loud():
    with pytest.raises(FileNotFoundError, match='unknown SAND provider'):
        _missing_sample(provider='bogus').resolve()


def test_download_missing_sand_is_loud(monkeypatch):
    def no_sand(*args, **kwargs):
        raise ImportError('No module named sand')

    monkeypatch.setattr(samples_module, 'import_module', no_sand)
    with pytest.raises(FileNotFoundError, match='"sand" package'):
        _missing_sample().resolve()


def test_download_provider_failure_is_file_not_found(monkeypatch, tmp_path):
    stub = _StubDownloader(error=RuntimeError('network down'))
    monkeypatch.setattr(samples_module, '_make_downloader', lambda p: stub)

    with pytest.raises(FileNotFoundError, match='PID.*network down') as exc:
        _missing_sample().resolve()
    assert 'test product' in str(exc.value)


def test_download_success_uses_cache_dir(monkeypatch, tmp_path):
    cache = tmp_path / 'cache'
    monkeypatch.setenv('EOREAD_SAMPLES_DIR', str(cache))
    stub = _StubDownloader()
    monkeypatch.setattr(samples_module, '_make_downloader', lambda p: stub)

    assert _missing_sample().resolve() == cache / 'PID'
    # the stub was asked to fetch PID into the cache root
    assert stub.calls == [('PID', cache)]


def test_download_missing_target_is_loud(monkeypatch, tmp_path):
    monkeypatch.setenv('EOREAD_SAMPLES_DIR', str(tmp_path))
    stub = _StubDownloader(skip_create=True)
    monkeypatch.setattr(samples_module, '_make_downloader', lambda p: stub)

    with pytest.raises(FileNotFoundError, match='missing target'):
        _missing_sample().resolve()


def test_download_default_cache_dir(monkeypatch):
    monkeypatch.delenv('EOREAD_SAMPLES_DIR', raising=False)
    assert samples_module._cache_dir() == Path.home() / '.cache' / 'eoread' / 'samples'
