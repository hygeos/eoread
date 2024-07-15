#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import contextmanager
import os
import shutil
import json
import getpass
import subprocess
import inspect

from typing import Optional, Union
from datetime import datetime
from functools import wraps
from tempfile import TemporaryDirectory
from time import sleep
from pathlib import Path
from eoread.utils.uncompress import uncompress as uncomp


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


@contextmanager
def LockFile(locked_file: Path,
             ext='.lock',
             interval=1,
             timeout=0,
             create_dir=True,
             ):
    """
    Create a blocking context with a lock file

    timeout: timeout in seconds, waiting to the lock to be released.
        If negative, disable lock files entirely.
    
    interval: interval in seconds

    Example:
        with LockFile('/dir/to/file.txt'):
            # create a file '/dir/to/file.txt.lock' including a filesystem lock
            # the context will enter once the lock is released
    """
    lock_file = Path(str(locked_file)+ext)
    disable = timeout < 0
    if create_dir and not disable:
        lock_file.parent.mkdir(exist_ok=True, parents=True)
    
    if disable:
        yield lock_file
    else:
        # wait untile the lock file does not exist anymore
        i = 0
        while lock_file.exists():
            if i > timeout:
                raise TimeoutError(f'Timeout on Lockfile "{lock_file}"')
            sleep(interval)
            i += 1

        # create the lock file
        with open(lock_file, 'w') as fd:
            fd.write('')

        try:
            yield lock_file
        finally:
            # remove the lock file
            lock_file.unlink()


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


class filegen:
    def __init__(self,
                 arg: Union[int, str]=0,
                 tmpdir: Optional[Path] = None,
                 lock_timeout: int = 0,
                 if_exists: str = 'skip',
                 uncompress: Optional[str] = None,
                 verbose: bool = True,
                 ):
        """
        A decorator for functions generating an output file.
        The path to this output file should is defined through `arg`.


        This decorator adds the following features to the function:
        - Use temporary file in a configurable directory, moved afterwards to final location
        - Detect existing file (if_exists='skip', 'overwrite', 'backup' or 'error')
        - Use output file lock when multiple functions may produce the same file
        The timeout for this lock is determined by argument `lock_timeout`.
        - Optional decompression
        
        Args:
            arg: int ot str (default 0)
                if int, defines the position of the positional argument defining the output file
                    (warning, starts at 1 for methods)
                if str, defines the argname of the keyword argument defining the output file
            tmpdir: which temporary directory to use
            lock_timeout (int): timeout in case of existing lock file
            if_exists (str): what to do in case of existing file
            uncompress (str): if specified, the wrapped function produces a file with the
                specified extension, typically '.zip'. This file is then uncompressed.
            verbose (bool): verbosity control

        Example:
            @filegen(arg=0)
            def f(path):
                open(path, 'w').write('test')
            f('/path/to/file.txt')

        Note:
            The arguments can be modified at runtime.
            Example:
                f.if_exists = 'overwrite'
                f.verbose = False
        """
        self._arg = arg
        self.tmpdir = tmpdir
        self.lock_timeout = lock_timeout
        self.if_exists = if_exists
        self.uncompress = uncompress
        self.verbose = verbose
        
    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if isinstance(self._arg, int):
                assert args, 'Error, no positional argument have been provided'
                assert (self._arg >= 0) and (self._arg < len(args))
                path = args[self._arg]
            elif isinstance(self._arg, str):
                assert self._arg in kwargs, \
                    f'Error, function should have keyword argument "{self._arg}"'
                path = kwargs[self._arg]
            else:
                raise TypeError(f'Invalid argumnt {self._arg}')
                
            ofile = Path(path)

            if skip(ofile, self.if_exists):
                if self.verbose:
                    print(f'Skipping existing file {ofile.name}')
                return
            
            with TemporaryDirectory(dir=self.tmpdir) as tmpd:
                
                # target (intermediary) file
                if self.uncompress:
                    tfile = Path(tmpd)/(ofile.name+self.uncompress)
                else:
                    tfile = Path(tmpd)/ofile.name

                with LockFile(ofile,
                            timeout=self.lock_timeout,
                            ):
                    if skip(ofile, self.if_exists):
                        return
                    if isinstance(self._arg, int):
                        updated_args = list(args)
                        updated_args[self._arg] = tfile
                        updated_kwargs = kwargs
                    elif isinstance(self._arg, str):
                        updated_args = args
                        updated_kwargs = {**kwargs, self._arg: tfile}
                    else:
                        raise ValueError(f'Invalid argumnt {self._arg}')
                        
                    ret = f(*updated_args, **updated_kwargs)
                    assert tfile.exists()

                    # check that the function does not return anything,
                    # because the function call may be skipped upon existing file
                    assert ret is None

                    if self.uncompress:
                        uncompressed = uncomp(tfile, Path(tmpd))
                        safe_move(uncompressed, ofile)
                    else:
                        safe_move(tfile, ofile)
            return
        return wrapper


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