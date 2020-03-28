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
from .uncompress import Uncompress


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


def download_url(product):
    """
    Download `product` from `url` to `dirname`

    If this is a zip file, uncompress it.
    """
    if 'url' not in product:
        return

    url = product['url']
    product_path = product['path']
    with TemporaryDirectory() as tmpdir:
        if 'archive' in product:
            name = product['archive']
        else:
            name = Path(url).name
        target = Path(tmpdir)/name

        cmd = f'wget {url} -O {target}'
        if subprocess.call(cmd.split()):
            raise Exception(f'Error running command "{cmd}"')
        assert target.exists()
        if Uncompress(target).is_archive():
            with Uncompress(target) as uncompressed:
                safe_move(uncompressed, product_path.parent)
        else:
            safe_move(target, product_path.parent)

        assert product_path.exists(), \
            f'{product_path} does not exist'

    return product_path


def download_sentinel(product):
    """
    Download a sentinel product
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

    product_path = product['path']

    api = SentinelAPI(**cred)
    with TemporaryDirectory() as tmpdir:
        api.download(pid, directory_path=tmpdir)

        zipfs = list(Path(tmpdir).iterdir())
        assert len(zipfs) == 1
        zipf = zipfs[0]

        # uncompress
        with Uncompress(zipf) as uncompressed:
            safe_move(uncompressed,
                      product_path.parent)

        assert product_path.exists(), \
            f'{product_path} does not exist'

    return product_path


def download(product):
    """
    Download a product from various sources
        - direct url
        - sentinel hubs

    Arguments:
    ----------

    product: dict with following keys:
        - 'path' local path (pathlib object)
          Example:
            - Path()/'S2A_MSIL1C_2019[...].SAFE',
            - 'S3A_OL_1_EFR____2019[...].SEN3'
        Download source (one or several)
        - 'scihub_id': '6271ae12-0e00-47d1-9a08-b0658d2262ad',
        - 'coda_id': 'a8c346c8-7752-4720-82b8-e5dea5fccf22',
        - 'url': 'https://earth.esa.int/[...]DLFE-451.zip',

    dirname: str
        base directory for download

    Returns: the path to the product
    """
    path = product['path']
    print(f'Getting {path}')

    if path.exists():
        print('Skipping existing product', path)
        return path

    p = download_url(product)
    if p:
        return p
    p = download_sentinel(product)
    if p:
        return p

    raise Exception(f'No valid method for retrieving product {path.name}')
