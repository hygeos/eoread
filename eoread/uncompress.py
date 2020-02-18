#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tempfile
from zipfile import ZipFile
from glob import glob
import tarfile
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
        self.zip_extensions = ['.zip']
        self.tar_extensions = ['.tar.gz', '.tgz', '.tar.bz2', '.tar']

    def __enter__(self):
        self.tmpdir = tempfile.TemporaryDirectory(dir=self.dirname)
        return self.uncompress(self.tmpdir.name)

    def is_archive(self):
        return True in [self.filename.endswith(fmt)
                        for fmt in self.zip_extensions + self.tar_extensions]

    def uncompress(self, dest):
        """
        Uncompresses the archive to dest, returns the path to the uncompressed file
        """
        if self.verbose:
            print(f'Uncompressing "{self.filename}" to {dest}')

        if True in [self.filename.endswith(fmt)
                    for fmt in self.zip_extensions]:
            # zip file
            with ZipFile(self.filename) as zipf:
                zipf.extractall(dest)

        elif True in [self.filename.endswith(fmt)
                      for fmt in self.tar_extensions]:
            # tar file
            with tarfile.open(self.filename) as tarf:
                tarf.extractall(path=dest)
        else:
            fname = os.path.basename(self.filename)
            raise Exception(f'Error handling extension for "{fname}"')

        glb = glob(os.path.join(dest, '*'))
        if len(glb) == 1:
            path = glb[0]
        else:
            path = dest

        return path


    def __exit__(self, typ, value, traceback):
        self.tmpdir.cleanup()
