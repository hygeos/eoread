#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities to download products
"""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from warnings import warn
from sentinelsat import SentinelAPI
from typing import Dict
from eoread.download_S2 import get_sentinel2_image
from eoread.utils.uncompress import uncompress as uncomp
from .common import timeit
from .utils.fileutils import filegen

from core import download, auth, ftp

def download_url(*args, **kwargs) -> Path:
    warn('Please use download_url from core.download')
    return download.download_url(*args, **kwargs)


def get_auth(name):
    warn('Please use get_auth from core.auth')
    return auth.get_auth(name)


def get_auth_dhus(name):
    warn('Please use get_auth_dhus from core.auth')
    return auth.get_auth_dhus(name)


def get_auth_ftp(name) -> Dict:
    warn('Please use get_auth_ftp from core.ftp')
    return ftp.get_auth_ftp(name)


def get_url_ftpfs(name):
    warn('Please use get_url_ftpfs from core.ftp')
    return ftp.get_url_ftpfs(name)


def download_sentinel(product, dirname):
    """
    Download a sentinel product to `dirname`
    """
    if 'scihub_id' in product:
        cred = get_auth('scihub')
        pid = product['scihub_id']
    elif 'coda_id' in product:
        cred = get_auth('coda')
        pid = product['coda_id']
    else:
        return None

    api = SentinelAPI(**cred)
    api.download(pid, directory_path=dirname)


@filegen()
def download_sentinelapi(target: Path,
                         source: str = 'scihub'):
    """
    Download a product using sentinelapi

    Source: scihub, coda
    """
    api = SentinelAPI(**get_auth_dhus(source))
    res = list(api.query(filename=target.name+'*'))
    assert len(res) == 1
    with TemporaryDirectory() as tmpdir:
        compressed = api.download(
            res[0],
            directory_path=tmpdir)
        uncompressed = uncomp(compressed['path'], tmpdir)
        shutil.move(uncompressed, target)


def download_multi(product):
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
        elif product['path'].name.startswith('S2'):
            import fels
            url = get_S2_google_url(product['path'].name)
            fels.get_sentinel2_image(url, tmpdir)
        elif ('scihub_id' in product) or ('coda_id' in product):
            download_sentinel(product, tmpdir)
        else:
            raise Exception(
                f'No valid method for retrieving product {path.name}')

        compressed = next(Path(tmpdir).glob('*'))

        uncomp(compressed, path.parent, on_uncompressed='copy')

    assert path.exists(), f'{path} does not exist.'

    return path


def get_S2_google_url(filename):
    """
    Get the google url for a given S2 product, like
    'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/32/T/QR/S2A_MSIL1C_20170111T100351_N0204_R122_T32TQR_20170111T100351.SAFE'

    The result of this function can be downloaded with the fels module:
        fels.get_sentinel2_image(url, directory)
    Note: the filename can be obtained either with the Sentinels hub api, or with
    google's catalog (see fels)
    """
    tile = filename.split('_')[-2]
    prod_type = filename.split('_')[1]
    assert len(tile) == 6
    utm = tile[1:3]
    pos0 = tile[3::4]
    pos1 = tile[4:]
    if prod_type == 'MSIL1C':
        url_base = 'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles'
    elif prod_type == 'MSIL2A':
        url_base = 'http://storage.googleapis.com/gcp-public-data-sentinel-2/L2/tiles'
        # gs://gcp-public-data-sentinel-2/L2/tiles/32/T/NS/S2B_MSIL2A_20210511T101559_N0300_R065_T32TNS_20210511T134528.SAFE/
    else:
        raise Exception(f'Unexpected product type "{prod_type}"')
    filename_full = filename if (filename.endswith('.SAFE')) else (filename+'.SAFE')
    url = f'{url_base}/{utm}/{pos0}/{pos1}/{filename_full}'

    return url

def download_S2_google(product, dirname, **kwargs) -> Path:
    target = Path(dirname)/(product+'.SAFE')
    @filegen(if_exists="skip", **kwargs)
    def down_S2(path):
        with timeit('Downloading'):
            print(f'Downloading {product}...')
            url = get_S2_google_url(product)
            get_sentinel2_image(url, outputdir=path.parent)
    down_S2(target)
    return target


def ftp_download(*args, **kwargs):
    warn('Please use ftp_download from core.ftp')
    return ftp.ftp_download(*args, **kwargs)


def ftp_file_exists(*args, **kwargs):
    warn('Please use ftp_file_exists from core.ftp')
    return ftp.ftp_file_exists(*args, **kwargs)


def ftp_create_dir(*args, **kwargs):
    warn('Please use ftp_create_dir from core.ftp')
    return ftp.ftp_create_dir(*args, **kwargs)


def ftp_upload(*args, **kwargs):
    warn('Please use ftp_upload from core.ftp')
    return ftp.ftp_upload(*args, **kwargs)


def ftp_list(*args, **kwargs):
    warn('Please use ftp_list from core.ftp')
    return ftp.ftp_list(*args, **kwargs)
