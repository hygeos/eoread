
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from eoread.utils.fileutils import LockFile, filegen, get_git_commit, mdir


def test_lockfile():
    with TemporaryDirectory() as tmpdir:
        lock_file = Path(tmpdir)/'test'
        with LockFile(lock_file) as lf:
            assert lf.exists()
            with pytest.raises(TimeoutError):
                with LockFile(lock_file):
                    pass
        with LockFile(lock_file) as lf:
            pass


@pytest.mark.parametrize('if_exists', ['skip', 'overwrite', 'backup'])
def test_filegen(if_exists):
    @filegen(arg=0, if_exists=if_exists)
    def f(path, target_path):
        with open(path, 'w') as fd:
            fd.write('test')
        
        # check that target file is currently locked
        with pytest.raises(TimeoutError):
            with LockFile(target_path):
                pass

    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)/'test'
        f(target, target)
        f(target, target)  # second call skips existing file


@pytest.mark.parametrize('if_exists', ['skip', 'overwrite', 'backup'])
def test_filegen_class(if_exists):
    class MyClass:
        def __init__(self) -> None:
            self.ncalls = 0
        @filegen(1, if_exists=if_exists)
        def method(self, path):
            self.ncalls += 1
            with open(path, 'w') as fd:
                fd.write('test')
    
    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)/'test'
        M = MyClass()
        M.method(target)
        assert M.ncalls == 1
        M.method(target)  # second call skips existing file
        assert M.ncalls == (1 if if_exists == 'skip' else 2)


def test_dirgen():
    """
    Test filegen on function that create directories
    """
    @filegen()
    def create_dir(directory: Path):
        mdir(directory)

    with TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir)/'test'
        create_dir(target_dir)
        assert target_dir.exists()


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
            mdir(Path(tmpdir), strict=True)


def test_get_git_commit():
    gc = get_git_commit()
    print(gc)
                    