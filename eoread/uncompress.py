#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from zipfile import ZipFile
import bz2
import gzip
import shutil
import tarfile
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def uncompress(filename,
               dirname,
               allow_uncompressed=True,
               create_out_dir=True):
    """
    Uncompress `filename` to `dirname`

    Arguments:
    ----------

    allow_uncompressed: bool
        if `filename` is not compressed, just move it to `dirname`
    create_out_dir: bool
        create output directory if it does not exist

    Returns the path to the uncompressed file
    """
    print(f'Uncompressing {filename}')
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
            print('Executing:', cmd)
            if subprocess.call(cmd.split()):
                raise Exception(f'Error executing command {cmd}')
            target_tmp = filename.parent/filename.stem
            assert target_tmp.exists()
        else:
            if allow_uncompressed:
                target_tmp = Path(filename)
            else:
                raise Exception(
                    'Could not determine format of file '
                    f'{Path(filename).name} and `allow_uncompressed` is not set.')

        # determine path to uncompressed temporary directory
        if target_tmp is None:
            lst = list(Path(tmpdir).glob('*'))
            assert len(lst) == 1
            target_tmp = lst[0]

        # determine target
        target = Path(dirname)/target_tmp.name
        assert not target.exists(), f'Error, {target} exists.'

        # move temporary to destination
        print(f'Moving uncompressed file to {target}')
        shutil.move(target_tmp, target)

    assert target.exists()

    return target