#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities to download products
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from textwrap import dedent
from .uncompress import uncompress
from .misc import LockFile, safe_move


def download_url(url, dirname, wget_opts='',
                 check_function=None, tmpdir=None,
                 if_exists='error',
                 lock_timeout=0, verbose=True):
    """
    Download `url` to `dirname`

    Uses a temporary directory `tmpdir`
    Options `wget_opts` are added to wget
    if_exists: 'error', 'skip' or 'overwrite'

    Returns the path to the downloaded file
    """
    target = Path(dirname)/(Path(url).name)
    lock = Path(dirname)/(Path(url).name+'.lock')
    if verbose:
        print('Downloading:', url)
        print('To: ', target)

    with LockFile(lock, timeout=lock_timeout), TemporaryDirectory(tmpdir) as tmpdir:
        tmpf = Path(tmpdir)/(Path(url).name+'.tmp')
        if (not target.exists()) or (target.exists() and (if_exists == 'overwrite')):

            cmd = f'wget {wget_opts} {url} -O {tmpf}'
            if subprocess.call(cmd.split()):
                raise Exception(f'Error running command "{cmd}"')
            assert tmpf.exists()

            if check_function is not None:
                check_function(tmpf)

            safe_move(tmpf, target)

            assert target.exists()
        
        elif if_exists == 'skip':
            print(f'Skipping existing file "{target}"')
        
        elif if_exists == 'error':
            raise Exception(f'Error, file {target} exists')

        else:
            raise ValueError(f'Error, invalid value for if_exists: "{if_exists}"')

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

    with TemporaryDirectory(prefix='tmp_eoread_download_') as tmpdir:
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


def get_S2_google_url(filename):
    """
    Get the google url for a given S2 product, like
    'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/32/T/QR/S2A_MSIL1C_20170111T100351_N0204_R122_T32TQR_20170111T100351.SAFE'

    The result of this function can be downloaded with the fels module:
        fels.get_sentinel2_image(url, directory)
    Note: the filename can be obtained either with the Sentinels hub api, or with google's catalog (see fels)
    """
    tile = filename.split('_')[-2]
    assert len(tile) == 6
    utm = tile[1:3]
    pos0 = tile[3::4]
    pos1 = tile[4:]
    url_base = 'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles'
    url = f'{url_base}/{utm}/{pos0}/{pos1}/{filename}'

    return url
