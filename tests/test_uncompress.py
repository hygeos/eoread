#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gzip
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from core.uncompress import CacheDir, duration


def test_uncompress_cache():
    with TemporaryDirectory(prefix='test_uncompress_cache') as tmpdir, \
            NamedTemporaryFile(suffix='.gz') as tmpfile:

        # create some compressed file
        with gzip.open(tmpfile, 'w') as fp:
            fp.write(b'Sample file')
        
        # uncompress a file twice
        # the returned path should be identical
        cdir = CacheDir(tmpdir)
        path1 = cdir.uncompress(tmpfile.name)
        path2 = cdir.uncompress(tmpfile.name)
        assert path1 == path2
        assert path1.exists()


def test_uncompress_uncompressed():
    # passing an uncompressed file should return the same file
    with TemporaryDirectory(prefix='test_uncompress_cache') as tmpdir, \
            NamedTemporaryFile() as tmpfile:
        # write a sample file
        with open(tmpfile.name, 'w') as fp:
            fp.write('Sample file')

        cdir = CacheDir(tmpdir)
        assert cdir.uncompress(tmpfile.name) == Path(tmpfile.name)
        

def test_duration():
    assert duration('2w') == timedelta(weeks=2)
    assert duration('2d') == timedelta(days=2)
    assert duration('2h') == timedelta(hours=2)
