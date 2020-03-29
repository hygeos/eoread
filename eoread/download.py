#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities to download products
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from textwrap import dedent
import shutil
from .uncompress import uncompress


def safe_move(src, dst, makedirs=True):
    """
    Move `src` file to `dst` directory

    if `makedirs`: create directory if necessary
    """
    pdst = Path(dst)
    if not pdst.exists():
        if makedirs:
            pdst.mkdir(parents=True)
        else:
            raise IOError(f'Error, directory {dst} does not exist')
    assert pdst.is_dir()
    psrc = Path(src)
    print(f'Moving "{src}" to "{dst}"...')
    target = pdst/psrc.name
    assert not target.exists()
    with TemporaryDirectory(prefix='copying_'+psrc.name+'_', dir=dst) as tmpdir:
        tmp = Path(tmpdir)/psrc.name
        shutil.move(psrc, tmp)
        shutil.move(tmp, target)


def download_url(url, dirname):
    """
    Download `url` to `dirname`

    Returns the path to the downloaded file
    """
    target = Path(dirname)/(Path(url).name)

    cmd = f'wget {url} -O {target}'
    if subprocess.call(cmd.split()):
        raise Exception(f'Error running command "{cmd}"')
    assert target.exists()

    return target


def download_sentinel(product, dirname):
    """
    Download a sentinel product to `dirname`
    """
    from sentinelsat import SentinelAPI
    try:
        import credentials
    except:
        raise Exception(dedent("""\
                        Could not import credentials module
                        Please create a file credentials.py with your credentials:
                        scihub = {'user': 'username',
                                  'password': 'password',
                                  'api_url': 'https://scihub.copernicus.eu/dhus/'}
                        coda = {'user': 'username',
                                'password': 'password',
                                'api_url': 'https://coda.eumetsat.int'}
                        """))

    if 'scihub_id' in product:
        cred = credentials.scihub
        pid = product['scihub_id']
    elif 'coda_id' in product:
        cred = credentials.coda
        pid = product['coda_id']
    else:
        return None

    api = SentinelAPI(**cred)
    api.download(pid, directory_path=dirname)


def download(product):
    """
    Download a product from various sources
        - direct url
        - sentinel hubs

    Arguments:
    ----------

    product: dict with following keys:
        - 'path' local target path (pathlib object)
          Example:
            - Path()/'S2A_MSIL1C_2019[...].SAFE',
            - 'S3A_OL_1_EFR____2019[...].SEN3'
        Download source (one or several)
        - 'scihub_id': '6271ae12-0e00-47d1-9a08-b0658d2262ad',
        - 'coda_id': 'a8c346c8-7752-4720-82b8-e5dea5fccf22',
        - 'url': 'https://earth.esa.int/[...]DLFE-451.zip',
    
    Uncompresses the downloaded product if necessary

    Returns: the path to the product
    """
    path = product['path']
    print(f'Getting {path}')

    if path.exists():
        print('Skipping existing product', path)
        return path

    with TemporaryDirectory() as tmpdir:
        if 'url' in product:
            download_url(product['url'], tmpdir)
        elif ('scihub_id' in product) or ('coda_id' in product):
            download_sentinel(product, tmpdir)
        else:
            raise Exception(
                f'No valid method for retrieving product {path.name}')

        compressed = next(Path(tmpdir).glob('*'))

        uncompress(compressed, path.parent)

    assert path.exists(), f'{path} does not exist.'

    return path
