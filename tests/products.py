#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import subprocess
import os
from sentinelsat import SentinelAPI
import zipfile
from glob import glob

'''
Get necessary products for testing
'''

try:
    import credentials
except:
    raise Exception("Could not import credentials module\n"
                    "Please create a file credentials.py with your coda credentials:\n"
                    "scihub = {'user': 'username',\n"
                    "          'password': 'password',\n"
                    "          'api_url': 'https://scihub.copernicus.eu/dhus/'}\n"
                    "coda = {'user': 'username',\n"
                    "        'password': 'password',\n"
                    "        'api_url': 'https://coda.eumetsat.int'}\n"
                    )

# products definition
prod_S2_L1_20190419 = {
    'id': '3ac99e56-8bff-4af3-b9bd-2e41a1ddaf61',
    'name': 'S2A_MSIL1C_20190419T105621_N0207_R094_T31UDS_20190419T130656.SAFE',
    'credentials': credentials.scihub
    }

prod_S3_L1_20190430 = {
    'id': '6271ae12-0e00-47d1-9a08-b0658d2262ad',
    'name': 'S3A_OL_1_EFR____20190430T094655_20190430T094955_20190501T131540_0179_044_136_2160_LN1_O_NT_002.SEN3',
    'credentials': credentials.scihub
    }

prod_S3_L2_20190612 = {
    'id': 'a8c346c8-7752-4720-82b8-e5dea5fccf22',
    'name': 'S3B_OL_2_WFR____20190612T085520_20190612T085820_20190613T175523_0179_026_221_2340_MAR_O_NT_002.SEN3',
    'credentials': credentials.coda,
    }

prod_meris_L1_20060822 = {
    # from https://earth.esa.int/web/guest/-/meris-sample-data-4320
    'url': 'https://earth.esa.int/c/document_library/get_file?folderId=23684&name=DLFE-451.zip',
    'name': 'MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1',
}

@pytest.fixture
def sample_data_path():
    p = 'SAMPLE_DATA'
    if not os.path.exists(p):
        os.makedirs(p)
    return p


@pytest.fixture
def sentinel_product(product, sample_data_path, capsys):
    '''
    Returns uncompressed Sentinel product
    '''
    prod_path = os.path.join(sample_data_path, product['name'])
    if os.path.exists(prod_path):
        return prod_path
    
    download_dir = os.path.join(sample_data_path, product['id'])
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    api = SentinelAPI(**product['credentials'])
    with capsys.disabled():
        print('Downloading', product['id'], 'under', product['name'])
        api.download(product['id'], directory_path=download_dir)
    
    zipf = glob(os.path.join(download_dir, '*'))[0]
    
    with zipfile.ZipFile(zipf, mode='r') as z, capsys.disabled():
        print('Uncompressing', zipf)
        z.extractall(path=sample_data_path)
    
    assert os.path.exists(prod_path)
    
    return prod_path


@pytest.fixture
def meris_product(product, sample_data_path, capsys):
    ''' Download and uncompress sample MERIS products '''
    target = os.path.join(sample_data_path, product['name'])
    if not os.path.exists(target):
        target_zip = os.path.join(sample_data_path, 'download_meris.zip')
        url = product['url']
        cmd = f'wget {url} -O {target_zip}'.split()
        with capsys.disabled():
            subprocess.call(cmd)

        with zipfile.ZipFile(target_zip, mode='r') as z, capsys.disabled():
            print('Uncompressing', target_zip)
            z.extractall(path=sample_data_path)

        os.remove(target_zip)

    return target
