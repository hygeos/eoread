#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import bz2
import gzip
import shutil
import tarfile
import subprocess
import json
import getpass

from pathlib import Path
from functools import wraps
from zipfile import ZipFile
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory, gettempdir, mkdtemp

from .fileutils import LockFile

class ErrorUncompressed(Exception):
    """
    Raised when input file is already not compressed
    """


def uncompress_decorator(filename='.eoread_uncompress_mapping',
                         verbose=True):
    """
    A decorator that uncompresses the result of function `f`
    
    Signature of `f` is assumed to be as follows:
        f(identifier, dirname, *args, **kwargs)
    
    The file returned by `f` is uncompressed to dirname
    
    The mapping of "identifier -> uncompressed" is stored in dirname/filename
    """
    def read_mapping(mapping_file: Path):
        if mapping_file.exists():
            with open(mapping_file) as fp:
                return json.load(fp)
        else:
            return {}
    def decorator(f):
        @wraps(f)
        def wrapper(identifier, dirname, *args, **kwargs):
            mapping_file = Path(dirname)/filename
            mapping = read_mapping(mapping_file)
            
            if identifier not in mapping:
                with TemporaryDirectory() as tmpdir:
                    f_compressed = f(identifier, tmpdir, *args, **kwargs)
                    target = uncompress(f_compressed, dirname, verbose=verbose)

                    with LockFile(mapping_file):
                        mapping = read_mapping(mapping_file)
                        mapping[identifier] = target.name
                        with open(mapping_file, 'w') as fp:
                            json.dump(mapping, fp, indent=4)
            else:
                target = Path(dirname)/mapping[identifier]
            assert target.exists()
            return target
        return wrapper
    return decorator


def uncompress(filename,
               dirname,
               on_uncompressed='error',
               create_out_dir=True,
               verbose=False) -> Path:
    """
    Uncompress `filename` to `dirname`

    Arguments:
    ----------

    on_uncompressed: str
        determines what to do if `filename` is not compressed
        - 'error': raise an error (default)
        - 'copy': copy uncompressed file
        - 'bypass': returns the input file
    create_out_dir: bool
        create output directory if it does not exist

    Returns the path to the uncompressed file
    """
    filename = Path(filename)
    if verbose:
        print(f'Uncompressing {filename} to {dirname}')
    if not Path(dirname).exists():
        if create_out_dir:
            Path(dirname).mkdir(parents=True)
        else:
            raise IOError(f'Directory {dirname} does not exist.')

    fname = str(filename)
    with TemporaryDirectory(prefix='tmp_uncompress',
                            dir=dirname) as tmpdir:

        # uncompress to temporary directory
        target_tmp = None
        if fname.endswith('.zip'):
            with ZipFile(fname) as zipf:
                zipf.extractall(tmpdir)

        elif True in [fname.endswith(x)
                      for x in ['.tar.gz', '.tgz', '.tar.bz2', '.tar']]:
            with tarfile.open(fname) as tarf:
                tarf.extractall(path=tmpdir)
        elif fname.endswith('.gz'):
            target_tmp = Path(tmpdir)/filename.stem
            with gzip.open(fname, 'rb') as f_in, open(target_tmp, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        elif fname.endswith('.bz2'):
            target_tmp = Path(tmpdir)/filename.stem
            with bz2.BZ2File(fname) as f_in, open(target_tmp, 'wb') as f_out:
                data = f_in.read()
                f_out.write(data)
        elif fname.endswith('.Z'):
            cmd = f'gunzip {fname}'
            if verbose:
                print('Executing:', cmd)
            if subprocess.call(cmd.split()):
                raise Exception(f'Error executing command {cmd}')
            target_tmp = filename.parent/filename.stem
            assert target_tmp.exists()
        else:
            if on_uncompressed == 'error':
                raise ErrorUncompressed(
                    'Could not determine format of file '
                    f'{Path(filename).name} and `allow_uncompressed` is not set.')
            elif on_uncompressed == 'copy':
                target_tmp = Path(filename)
            elif on_uncompressed == 'bypass':
                return filename
            else:
                raise ValueError(f'Invalid value "{on_uncompressed}" for argument `on_uncompressed`')

        # determine path to uncompressed temporary directory and target
        target = None
        if target_tmp is None:
            lst = list(Path(tmpdir).glob('*'))
            if len(lst) == 1:
                target_tmp = lst[0]
            else:
                target_tmp = Path(tmpdir)
                target = Path(dirname)/filename.stem

        # determine target
        if target is None:
            target = Path(dirname)/target_tmp.name
        assert not target.exists(), f'Error, {target} exists.'

        # move temporary to destination
        shutil.move(target_tmp, target)

    assert target.exists()

    return target


def now_isofmt():
    """
    Returns now in iso format
    """
    return datetime.now().isoformat()

def duration(s):
    """
    Returns a timedelta from a string `s`
    """
    if s.endswith('w'):
        return timedelta(weeks=float(s[:-1]))
    elif s.endswith('d'):
        return timedelta(days=float(s[:-1]))
    elif s.endswith('h'):
        return timedelta(hours=float(s[:-1]))
    else:
        raise Exception(f'Can not convert "{s}"')


class CacheDir:
    """
    A cache directory for uncompressing files

    Example:
        # by default, CacheDir stores data in /tmp/uncompress_cache_<user>
        uncompressed = CacheDir().uncompress(compressed_file)
    """
    def __init__(self, directory=None):
        directory = directory or Path(gettempdir())/f'uncompress_cache_{getpass.getuser()}'
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
        self.readme_path = self.directory/'README.TXT'
        self.prefix = 'cache_'
        self.info_file = 'info.json'

        # initialize readme file
        if self.readme_path.exists():
            assert self.readme_path.is_file()
        else:
            with open(self.readme_path, 'w') as fp:
                fp.write(f'This directory was created by {__file__} on {datetime.now()}')

    def read_info(self, directory):
        info_file = directory/self.info_file
        with open(info_file) as fp:
            info = json.load(fp)
        return info
    
    def write_info(self, directory, info):
        info_file = directory/self.info_file
        with TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir)/info_file.name
            with open(tmpfile, 'w') as fp:
                json.dump(info, fp, indent=4)
            shutil.move(tmpfile, info_file)

    def find(self, file_compressed):
        '''
        Finds the directory containing `file_compressed`
        and returns the related uncompressed file (or None)
        '''
        file_uncompressed = None
        for d in self.directory.glob(self.prefix+'*'):
            info = self.read_info(d)
            if Path(file_compressed).resolve() == Path(info['path_compressed']):
                file_uncompressed = info['path_uncompressed']
                info['accessed'] = now_isofmt()
                self.write_info(d, info)
            
            # check if file must be purged
            accessed = datetime.fromisoformat(info['accessed'])
            purge_after = duration(info['purge_after'])
            if accessed + purge_after < datetime.now():
                shutil.rmtree(d)
        
        return file_uncompressed


    def uncompress(self, filename, purge_after='1w'):
        filename = Path(filename)
        uncompressed = self.find(filename)

        if uncompressed is None:
            with TemporaryDirectory(dir=self.directory, prefix='uncompressing_') as tmpdir:
                try:
                    tmp_uncompressed = uncompress(filename, tmpdir)
                except ErrorUncompressed:
                    return Path(filename)
            
                # check that no other thread uncompressed the same file meanwhile
                uncompressed = self.find(filename)
                if uncompressed is None:
                    # create the new directory
                    directory = Path(mkdtemp(dir=self.directory,
                                            prefix=self.prefix+'_'+filename.name+'_'))
                    uncompressed = directory/tmp_uncompressed.name

                    info = {
                        'path_compressed': str(filename),
                        'path_uncompressed': str(uncompressed),
                        'purge_after': purge_after,
                        'created': now_isofmt(),
                        'accessed': now_isofmt(),
                    }
                    self.write_info(directory, info)

                    shutil.move(tmp_uncompressed, uncompressed)
        
        return Path(uncompressed)
