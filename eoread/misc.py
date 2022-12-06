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


def only(x, description=None):
    """
    Small utility function to get the element of a single-element list
    """
    x = list(x)
    assert len(x) == 1, f'Error in {description}'
    return x[0] 


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
                 disable=False,
                ):
        self.lock_file = Path(str(lock_file)+ext)
        if create_dir and not disable:
            self.lock_file.parent.mkdir(exist_ok=True, parents=True)
        self.fd = None
        self.interval = interval
        self.timeout = timeout
        self.disable = disable

    def __enter__(self):
        if self.disable:
            return
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
        if self.disable:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.flush()
            self.fd.close()
            remove(self.lock_file)
        except FileNotFoundError:
            pass


class PersistentList(list):
    """
    A list that saves its content in `filename` on each modification. The extension
    must be `.json`.
    
    `concurrent`: whether to activate concurrent mode. In this mode, the
        file is also read before each access.
    """
    def __init__(self, filename, concurrent=True):
        self._filename = Path(filename)
        self.concurrent = concurrent
        assert str(filename).endswith('.json')
        self._read()

        # use `_autosave` decorator on all of these methods
        for attr in ('append', 'extend', 'insert', 'pop',
                     'remove', 'reverse', 'sort', 'clear'):
            setattr(self, attr, self._autosave(getattr(self, attr)))
        
        # trigger read on all of these methods
        for attr in ('__getitem__',):
            setattr(self, attr, self._autoread(getattr(self, attr)))

    def __len__(self):
        # for some reason, len() does not work with _autoread wrapper
        if self.concurrent:
            self._read()
        return list.__len__(self)

    def _autoread(self, func):
        @wraps(func)
        def _func(*args, **kwargs):
            if self.concurrent:
                self._read()
            return func(*args, **kwargs)
        return _func

    def _autosave(self, func):
        @wraps(func)
        def _func(*args, **kwargs):
            with LockFile(self._filename, disable=(not self.concurrent)):
                if self.concurrent:
                    self._read()
                ret = func(*args, **kwargs)
                self._save()
                return ret
        return _func

    def _read(self):
        list.clear(self)
        if self._filename.exists():
            with open(self._filename) as fp:
                # don't call .extend directly, as it would
                # recursively trigger read and save
                list.extend(self, json.load(fp))

    def _save(self):
        with open(self._filename, 'w') as fp:
            json.dump(self.copy(), fp, indent=4)


def skip(filename, if_exists='skip'):
    """
    Utility function to check whether a file should be skipped
    """
    if Path(filename).exists():
        if if_exists == 'skip':
            return True
        elif if_exists == 'error':
            raise FileExistsError(f'File {filename} exists.')
        else:
            raise ValueError(f'Invalid argument if_exists={if_exists}')
    else:
        return False


def filegen(arg=0,
            lock_timeout=0,
            tmpdir=None,
            if_exists='skip',
            check_return_none=True,
            ):
    """
    A decorator for functions generating an output file.
    The path to this output file should be provided to the function under
    named argument `path`.

    This decorator adds the following features to the function:
    - Use temporary file in a configurable directory, moved afterwards to final location
    - Detect existing file (if_exists='skip' or 'error')
    - Use output file lock when multiple functions may produce the same file
      The timeout for this lock is determined by argument `lock_timeout`.
    
    argname: int ot str (default 0)
        if int, defines the position of the positional argument defining the output file
            (warning, starts at 1 for methods)
        if str, defines the argname of the keyword argument defining the output file

    Example:
        @filegen()
        def f(path):
            open(path, 'w').write('test')
        f(path='/path/to/file.txt')
    
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if isinstance(arg, int):
                assert args, 'Error, no positional argument have been provided'
                assert (arg >= 0) and (arg < len(args))
                path = args[arg]
            elif isinstance(arg, str):
                assert arg in kwargs, \
                    f'Error, function should have keyword argument "{arg}"'
                path = kwargs[arg]
            else:
                raise ValueError(f'Invalid argumnt {arg}')
                
            ofile = Path(path)

            if skip(ofile, if_exists):
                return
            
            with TemporaryDirectory(dir=tmpdir) as tmpd:
                tfile = Path(tmpd)/ofile.name
                with LockFile(ofile,
                              timeout=lock_timeout,
                              ):
                    if skip(ofile, if_exists):
                        return
                    if isinstance(arg, int):
                        updated_args = list(args)
                        updated_args[arg] = tfile
                        updated_kwargs = kwargs
                    elif isinstance(arg, str):
                        updated_args = args
                        updated_kwargs = {**kwargs, arg: tfile}
                    else:
                        raise ValueError(f'Invalid argumnt {arg}')
                        
                    ret = f(*updated_args, **updated_kwargs)
                    assert tfile.exists()
                    if check_return_none:
                        # the function should not return anything,
                        # because it may be skipped
                        assert ret is None
                    safe_move(tfile, ofile)
            return
        return wrapper
    return decorator

