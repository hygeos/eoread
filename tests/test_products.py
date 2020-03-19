#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Define and download test products defined in products.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from textwrap import dedent
import shutil
import pytest
from eoread.uncompress import Uncompress
from tests.products import products, get_path, dir_samples, get_dir


@pytest.mark.parametrize('product', products.values(),
                         ids=list(products.keys()))
def test_available(product):
    path = get_path(product)
    if not path.exists():
        raise Exception(
            f'{path} is missing. '
            'You may run `python -m tests.test_products` to download.')


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


def download_url(product, dirname):
    """
    Download `product` from `url` to `dirname`

    If this is a zip file, uncompress it.
    """
    if 'url' not in product:
        return

    url = product['url']
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
                safe_move(uncompressed, dirname)
        else:
            safe_move(target, dirname)

    return 1


def download_sentinel(product, dirname):
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

    api = SentinelAPI(**cred)
    with TemporaryDirectory() as tmpdir:
        api.download(pid, directory_path=tmpdir)

        zipfs = list(Path(tmpdir).iterdir())
        assert len(zipfs) == 1
        zipf = zipfs[0]

        # uncompress
        with Uncompress(zipf) as uncompressed:
            safe_move(uncompressed, dirname)

        # safe_move(zipf, dir_samples)

    return 1


def download(product):
    name = product['name']
    print(f'Getting {name}')
    assert Path(dir_samples).exists(), \
        f'{dir_samples} does not exist. Please create it or link it before proceeding.'

    if get_path(product).exists():
        print(f'Skipping existing product {name}')
        return

    if download_url(product, get_dir(product)):
        return
    if download_sentinel(product, get_dir(product)):
        return

    raise Exception(f'No valid method for retrieving product {name}')

if __name__ == "__main__":
    print('Downloading sample products...')
    for k, v in products.items():
        download(v)
