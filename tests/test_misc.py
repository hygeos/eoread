
from functools import partial
import pytest
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import sleep
from eoread.misc import LockFile

def f(file_lock, interval):
    print(f'Started at {datetime.now()}')
    with LockFile(file_lock):
        print('Lock acquired at', datetime.now())
        sys.stdout.flush()
        sleep(interval)
        print('Lock released at', datetime.now())
        sys.stdout.flush()

@pytest.mark.parametrize('interval', [0, 0.1, 1])
@pytest.mark.parametrize('dirname', [None, '/rfs/tmp/', '/home/francois/tmp/'])
def test_lockfile(dirname, interval):
    with TemporaryDirectory(dir=dirname) as tmpdir:
        lock_file = Path(tmpdir)/'test'
        Pool(4).map(partial(f, interval=interval), [lock_file]*4)
        assert not lock_file.exists()
