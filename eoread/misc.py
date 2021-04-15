#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from os import remove, system
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import sleep
import json
from functools import wraps


def safe_move(src, dst, makedirs=True):
    """
    Move `src` file to `dst`

    if `makedirs`: create directory if necessary
    """
    pdst = Path(dst)
    psrc = Path(src)

    if pdst.exists():
        raise IOError(f'Error, {dst} exists')
    if not pdst.parent.exists():
        if makedirs:
            pdst.parent.mkdir(parents=True)
        else:
            raise IOError(f'Error, directory {pdst.parent} does not exist')
    print(f'Moving "{psrc}" to "{pdst}"...')

    with TemporaryDirectory(prefix='copying_'+psrc.name+'_', dir=pdst.parent) as tmpdir:
        tmp = Path(tmpdir)/psrc.name
        shutil.move(psrc, tmp)
        shutil.move(tmp, pdst)

    assert pdst.exists()


class LockFile:
    '''
    Create a context with a lock file

    Wait until the file is removed, to proceed.
    Designed to avoid concurrent processes simultaneously accessing a file.
    '''
    def __init__(self, lock_file, timeout=3600):
        self.lock_file = Path(lock_file)
        self.lock_file.parent.mkdir(exist_ok=True, parents=True)
        self.timeout = timeout

    def __enter__(self):
        if self.lock_file.exists():
            print(f'Lock file {self.lock_file} is present, waiting...')
        i = 0
        while self.lock_file.exists():
            if i >= self.timeout:
                raise Exception(f'Error, cannot acquire lock "{self.lock_file}"')

            sleep(1)
            i += 1


        system('touch {}'.format(self.lock_file))

    def __exit__(self, type, value, traceback):
        remove(self.lock_file)


class PersistentList(list):
    """
    A list that saves its content in `filename` on each modification
    """
    def __init__(self, filename):
        self._filename = Path(filename)
        assert str(filename).endswith('.json')

        if self._filename.exists():
            with open(self._filename) as fp:
                self.extend(json.load(fp))

        # trigger save on all of these methods
        for attr in ('append', 'extend', 'insert', 'pop', 'remove', 'reverse', 'sort'):
            setattr(self, attr, self._autosave(getattr(self, attr)))

    def _autosave(self, func):
        @wraps(func)
        def _func(*args, **kwargs):
            ret = func(*args, **kwargs)
            self._save()
            return ret
        return _func

    def _save(self):
        with open(self._filename, 'w') as fp:
            json.dump(self, fp, indent=4)
