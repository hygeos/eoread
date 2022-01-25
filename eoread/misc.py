#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime
from os import remove, system
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import sleep
import json
from functools import wraps
import fcntl


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
    """
    Create a blocking context with a lock file

    Ex:
    with LockFile('/dir/to/file.txt'):
        # create a file '/dir/to/file.txt.lock' including a filesystem lock
        # the context will enter once the lock is released
    """
    def __init__(self,
                 lock_file,
                 ext='.lock',
                 interval=1,
                 timeout=600,
                 create_dir=True,
                ):
        self.lock_file = Path(str(lock_file)+ext)
        if create_dir:
            self.lock_file.parent.mkdir(exist_ok=True, parents=True)
        self.fd = None
        self.interval = interval
        self.timeout = timeout

    def __enter__(self):
        i = 0
        while True:
            self.fd = open(self.lock_file, 'w')
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.fd.flush()
                break
            except (BlockingIOError, FileNotFoundError):
                self.fd.close()
                sleep(self.interval)
                i += 1
                if i > self.timeout:
                    raise TimeoutError(f'Timeout on Lockfile "{self.lock_file}"')

    def __exit__(self, type, value, traceback):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.flush()
            self.fd.close()
            remove(self.lock_file)
        except FileNotFoundError:
            pass


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


def skip(filename, on_exist='skip'):
    """
    Utility function to check whether a file should be skipped
    """
    if Path(filename).exists():
        if on_exist == 'skip':
            return True
        elif on_exist == 'error':
            raise IOError(f'File {filename} exists.')
        else:
            raise ValueError(f'Invalid argument on_exist={on_exist}')
    else:
        return False


def filegen(lock_timeout=600,
            tmpdir=None,
            on_exist='skip',
            varname='path',
            ):
    """
    A decorator for functions generating an output file.
    The path to this output file should be provided to the function under
    named argument `path`.

    This decorator adds the following features to the function:
    - Use temporary file in a configurable directory, moved afterwards to final location
    - Detect existing file (on_exist='skip' or 'error')
    - Use output file lock when multiple functions may produce the same file
      The timeout for this lock is determined by argument `lock_timeout`.

    Example:
        @filegen()
        def f(path):
            open(path, 'w').write('test')
        f(path='/path/to/file.txt')
    
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            assert varname in kwargs, \
                f'Error, function should have keyword argument "{varname}"'
            path = kwargs[varname]
            ofile = Path(path)

            if skip(ofile, on_exist):
                return
            
            with TemporaryDirectory(tmpdir) as tmpd:
                tfile = Path(tmpd)/ofile.name
                with LockFile(ofile,
                              timeout=lock_timeout,
                              ):
                    if skip(ofile, on_exist):
                        return
                    updated_kwargs = {**kwargs, varname: tfile}
                    ret = f(*args, **updated_kwargs)
                    assert tfile.exists()
                    assert ret is None   # the function should not return anything,
                                         # because it may be skipped
                    safe_move(tfile, ofile)
            return
        return wrapper
    return decorator

