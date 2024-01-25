#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import json
import fcntl
import getpass
import subprocess
import inspect

from os import remove
from typing import Union
from datetime import datetime
from functools import wraps
from tempfile import TemporaryDirectory
from time import sleep
from pathlib import Path

cfg = {
    # module-wide configuration
    'lock_timeout': 0,
    'tmpdir': None,
    'if_exists': 'skip',
}


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

    timeout: timeout in seconds, waiting to the lock to be released.
        If negative, disable lock files entirely.

    Ex:
    with LockFile('/dir/to/file.txt'):
        # create a file '/dir/to/file.txt.lock' including a filesystem lock
        # the context will enter once the lock is released
    """
    def __init__(self,
                 lock_file,
                 ext='.lock',
                 interval=1,
                 timeout=0,
                 create_dir=True,
                ):
        self.lock_file = Path(str(lock_file)+ext)
        if create_dir and (timeout >= 0):
            self.lock_file.parent.mkdir(exist_ok=True, parents=True)
        self.fd = None
        self.interval = interval
        self.timeout = timeout
        self.disable = timeout < 0

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
    def __init__(self,
                 filename,
                 timeout=0,
                 concurrent=True):
        self._filename = Path(filename)
        self.concurrent = concurrent
        self.timeout = timeout
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
            with LockFile(self._filename,
                          timeout=self.timeout):
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
        tmpfile = self._filename.parent/(self._filename.name+'.tmp')
        with open(tmpfile, 'w') as fp:
            json.dump(self.copy(), fp, indent=4)
        shutil.move(tmpfile, self._filename)


def skip(filename: Path,
         if_exists: str='skip'):
    """
    Utility function to check whether to skip an existing file
    
    if_exists:
        'skip': skip the existing file
        'error': raise an error on existing file
        'overwrite': overwrite existing file
        'backup': move existing file to a backup '*.1', '*.2'...
    """
    if Path(filename).exists():
        if if_exists == 'skip':
            return True
        elif if_exists == 'error':
            raise FileExistsError(f'File {filename} exists.')
        elif if_exists == 'overwrite':
            os.remove(filename)
            return False
        elif if_exists == 'backup':
            i = 0
            while True:
                i += 1
                file_backup = str(filename)+'.'+str(i)
                if not Path(file_backup).exists():
                    break
                if i >= 100:
                    raise FileExistsError()
            shutil.move(filename, file_backup)
        else:
            raise ValueError(f'Invalid argument if_exists={if_exists}')
    else:
        return False


def filegen(arg: Union[int, str]=0,
            check_return_none=True,
            **fg_kwargs
            ):
    """
    A decorator for functions generating an output file.
    The path to this output file should is defined through `arg`.

    This decorator adds the following features to the function:
    - Use temporary file in a configurable directory, moved afterwards to final location
    - Detect existing file (if_exists='skip', 'overwrite', 'backup' or 'error')
    - Use output file lock when multiple functions may produce the same file
      The timeout for this lock is determined by argument `lock_timeout`.
    
    arg: int ot str (default 0)
        if int, defines the position of the positional argument defining the output file
            (warning, starts at 1 for methods)
        if str, defines the argname of the keyword argument defining the output file

    Example:
        @filegen()
        def f(path):
            open(path, 'w').write('test')
        f(path='/path/to/file.txt')
    
    Configuration arguments can be passed to filegen(), or to the wrapped function,
    or modified module-wise through the 'cfg' dictionary.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, filegen_kwargs=None, **kwargs):
            # configuration: take first module_wide configuration,
            # then filegen_kwargs
            config = {**cfg,
                      **fg_kwargs,
                      **(filegen_kwargs or {})}
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

            if skip(ofile, config['if_exists']):
                return
            
            with TemporaryDirectory(dir=config['tmpdir']) as tmpd:
                tfile = Path(tmpd)/ofile.name
                with LockFile(ofile,
                              timeout=config['lock_timeout'],
                              ):
                    if skip(ofile, config['if_exists']):
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


def get_git_commit():
    try:
        return subprocess.check_output(
            ['git', 'describe', '--always', '--dirty']).decode()[:-1]
    except subprocess.CalledProcessError:
        return '<could not get git commit>'


def mdir(directory: Union[Path,str],
         mdir_filename: str='mdir.json',
         strict: bool=False,
         create: bool=True,
         **kwargs
         ) -> Path:
    """
    Create or access a managed directory with path `directory`
    Returns the directory path, so that it can be used in directories definition:
        dir_data = mdir('/path/to/data/')

    tag it with a file `mdir.json`, containing:
        - The creation date
        - The last access date
        - The python file and module that was run during access
        - The username
        - The current git commit if available
        - Any other kwargs, such as:
            - project
            - version
            - description
            - etc

    mdir_filename: default='mdir.json'

    strict: boolean
        False: metadata is updated
        True: metadata is checked or added (default)
           (remove file content to override)

    create: whether directory is automatically created (default True)
    """
    d = Path(directory)
    mdir_file = d/mdir_filename

    caller = inspect.stack()[1]

    # Attributes to check
    attrs = {
        'caller_file': caller.filename,
        'caller_function': caller.function,
        'git_commit': get_git_commit(),
        'username': getpass.getuser(),
        **kwargs,
    }

    data_init = {
        '__comment__': 'This file has been automatically created '
                        'by mdir() upon managed directory creation, '
                        'and stores metadata.',
        'creation_date': str(datetime.now()),
        **attrs,
    }

    modified = False
    if not d.exists():
        if not create:
            raise FileNotFoundError(
                f'Directory {d} does not exist, '
                'please create it [mdir(..., create=False)]')
        d.mkdir(parents=True)
        data = data_init
        modified = True
    else:
        if not mdir_file.exists():
            if strict:
                raise FileNotFoundError(
                    f'Directory {d} has been wrapped by mdir '
                    'but does not contain a mdir file.')
            else:
                data = data_init
                modified = True
        else:
            with open(mdir_file, encoding='utf-8') as fp:
                data = json.load(fp)

        for k, v in attrs.items():
            if k in data:
                if v != data[k]:
                    if strict:
                        raise ValueError(f'Mismatch of "{k}" in {d} ({v} != {data[k]})')
                    else:
                        data[k] = v
                        modified = True
            else:
                data[k] = v
                modified = True

    if modified:
        with open(mdir_file, 'w', encoding='utf-8') as fp:
            json.dump(data_init, fp, indent=4)

    return d