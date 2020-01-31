#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tempfile
from zipfile import ZipFile
from glob import glob
import os

class Uncompress:
    """
    Uncompresses a file in a temporary location

    Used as a context manager, it returns the path to the uncompressed file.
    If there is a single directory in the archive, this path is to that directory.

    Example:
        with Uncompress('file.zip') as unzipped:
            unzipped  # the path to the uncompressed data
    """
    def __init__(self,
                 filename,
                 dirname=None,
                 verbose=True):
        self.filename = filename
        self.tmpdir = None
        self.dirname = dirname
        self.verbose = verbose

    def __enter__(self):
        self.tmpdir = tempfile.TemporaryDirectory(dir=self.dirname)
        if self.verbose:
            print(f'Uncompressing "{self.filename}"')

        with ZipFile(self.filename) as zipf:
            zipf.extractall(self.tmpdir.name)

        glb = glob(os.path.join(self.tmpdir.name, '*'))
        if len(glb) == 1:
            path = glb[0]
        else:
            path = self.tmpdir.name

        return path

    def __exit__(self, typ, value, traceback):
        self.tmpdir.cleanup()
