
from functools import partial
import pytest
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import sleep
from eoread.fileutils import LockFile, filegen, get_git_commit, mdir

def f(file_lock, interval):
    print(f'Started at {datetime.now()}')
    with LockFile(file_lock, timeout=10):
        print('Lock acquired at', datetime.now())
        sys.stdout.flush()
        sleep(interval)
        print('Lock released at', datetime.now())
        sys.stdout.flush()

@pytest.mark.parametrize('interval', [0, 0.1, 1])
def test_lockfile(interval):
    with TemporaryDirectory() as tmpdir:
        lock_file = Path(tmpdir)/'test'
        Pool(4).map(partial(f, interval=interval), [lock_file]*4)
        assert not lock_file.exists()


@pytest.mark.parametrize('if_exists', ['skip', 'overwrite', 'backup'])
def test_filegen(if_exists):
    @filegen(if_exists=if_exists)
    def f(path):
        with open(path, 'w') as fd:
            fd.write('test')

    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)/'test'
        f(target)
        f(target)  # second call skips existing file


@pytest.mark.parametrize('if_exists', ['skip', 'overwrite', 'backup'])
def test_filegen_class(if_exists):
    class MyClass:
        @filegen(1, if_exists=if_exists)
        def method(self, path):
            with open(path, 'w') as fd:
                fd.write('test')
    
    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)/'test'
        M = MyClass()
        M.method(target)
        M.method(target)  # second call skips existing file


def test_mdir():
    with TemporaryDirectory() as tmpdir:
        for _ in range(2):
            mdir(Path(tmpdir)/'managed_directory',
                 description='A sample managed directory')


def test_missing_mdir():
    """
    Check that mdir fails when a mdir.json does not exist
    in an existing directory
    """
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            mdir(Path(tmpdir))


def test_get_git_commit():
    gc = get_git_commit()
    print(gc)
                    