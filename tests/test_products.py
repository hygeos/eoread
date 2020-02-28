#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Define and download test products defined in products.py
"""

import os
from tempfile import TemporaryDirectory
import subprocess
from glob import glob
from textwrap import dedent
import shutil
import pytest
from eoread.uncompress import Uncompress


dir_samples = 'SAMPLE_DATA'

# definition of test products
# - name, product: path to the product
# - scihub_id, coda_id: key for downloading on scihub/coda
# - url: url for direct download
# - archive: basename of the downloaded file (defaults to the basename of 'url')
products = {
    'prod_S2_L1_20190419': {
        'name': 'S2A_MSIL1C_20190419T105621_N0207_R094_T31UDS_20190419T130656.SAFE',
        'folder': 'MSI',
        'scihub_id': '3ac99e56-8bff-4af3-b9bd-2e41a1ddaf61',
        'url': 'http://download.hygeos.com/EOREAD_TESTDATA/MSI/S2A_MSIL1C_20190419T105621_N0207_R094_T31UDS_20190419T130656.zip',
    },
    'prod_S3_L1_20190430': {
        'name': 'S3A_OL_1_EFR____20190430T094655_20190430T094955_20190501T131540_0179_044_136_2160_LN1_O_NT_002.SEN3',
        'folder': 'OLCI',
        'scihub_id': '6271ae12-0e00-47d1-9a08-b0658d2262ad',
        'url': 'http://download.hygeos.com/EOREAD_TESTDATA/OLCI/S3A_OL_1_EFR____20190430T094655_20190430T094955_20190501T131540_0179_044_136_2160_LN1_O_NT_002.zip',
    },
    'prod_S3_L2_20190612': {
        'name': 'S3B_OL_2_WFR____20190612T085520_20190612T085820_20190613T175523_0179_026_221_2340_MAR_O_NT_002.SEN3',
        'folder': 'OLCI',
        'coda_id': 'a8c346c8-7752-4720-82b8-e5dea5fccf22',
        'url': 'http://download.hygeos.com/EOREAD_TESTDATA/OLCI/S3B_OL_2_WFR____20190612T085520_20190612T085820_20190613T175523_0179_026_221_2340_MAR_O_NT_002.zip',
    },
    'prod_meris_L1_20060822': {
        # from https://earth.esa.int/web/guest/-/meris-sample-data-4320
        'name': 'MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1',
        'folder': 'MERIS',
        'url': 'https://earth.esa.int/c/document_library/get_file?folderId=23684&name=DLFE-451.zip',
        'archive': 'MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1.zip',
    },
    'prod_oli_L1': {
        'name': 'LC80140282017275LGN00',
        'folder': 'LANDSAT8_OLI',
        'url': 'http://download.hygeos.com/EOREAD_TESTDATA/LANDSAT8_OLI/LC80140282017275LGN00.tar.gz',
    },
    'prod_sgli': {
        'name': 'GC1SG1_201912050159F05712_1BSG_VNRDQ_1007.h5',
        'folder': 'SGLI',
        'url': 'http://download.hygeos.com/EOREAD_TESTDATA/SGLI/'
               'GC1SG1_201912050159F05712_1BSG_VNRDQ_1007.h5',
    },
}


def get_path(product):
    """ Returns the path to the product """
    return os.path.join(dir_samples, product['folder'], product['name'])

def get_dir(product):
    """ Returns the directory containing the product """
    return os.path.join(dir_samples, product['folder'])


@pytest.mark.parametrize('product', products.values(),
                         ids=list(products.keys()))
def test_available(product):
    path = get_path(product)
    if not os.path.exists(path):
        raise Exception(
            f'{path} is missing. '
            'You may run `python -m tests.test_products` to download.')


def safe_move(src, dst, makedirs=True):
    """
    Move `src` file to `dst` directory

    if `makedirs`: create directory if necessary
    """
    if not os.path.exists(dst):
        if makedirs:
            os.makedirs(dst)
        else:
            raise IOError(f'Error, directory {dst} does not exist')
    assert os.path.isdir(dst)
    print(f'Moving "{src}" to "{dst}"...')
    basename = os.path.basename(src)
    target = os.path.join(dst, basename)
    with TemporaryDirectory(prefix='copying_'+basename+'_', dir=dst) as tmpdir:
        tmp = os.path.join(tmpdir, basename)
        shutil.move(src, tmp)
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
            name = os.path.basename(url)
        target = os.path.join(
            tmpdir,
            name)

        cmd = f'wget {url} -O {target}'
        if subprocess.call(cmd.split()):
            raise Exception(f'Error running command "{cmd}"')
        assert os.path.exists(target)
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

        zipfs = glob(os.path.join(tmpdir, '*'))
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
    assert os.path.exists(dir_samples), \
        f'{dir_samples} does not exist. Please create it or link it before proceeding.'

    if os.path.exists(get_path(product)):
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
