#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
from sentinelsat import SentinelAPI
import zipfile
from glob import glob

'''
Get necessary products for testing
'''

# products definition
prod_S2_20190419 = (
    '3ac99e56-8bff-4af3-b9bd-2e41a1ddaf61',
    'S2A_MSIL1C_20190419T105621_N0207_R094_T31UDS_20190419T130656.SAFE')

prod_S3_20190430 = (
    '505b7adf-95c9-42e3-a988-6909e65cbc7f',
    'S3A_OL_1_EFR____20190430T094655_20190430T094955_20190430T114141_0180_044_136_2160_LN1_O_NR_002.SEN3')

try:
    import credentials
except:
    raise Exception("Could not import credentials module\n"
                    "Please create a file credentials.py with your coda credentials:\n"
                    "scihub = {'user': 'username',\n"
                    "          'password': 'password',\n"
                    "          'api_url': 'https://scihub.copernicus.eu/dhus/'}\n"
                    )


@pytest.fixture
def sample_data_path():
    p = 'SAMPLE_DATA'
    if not os.path.exists(p):
        os.makedirs(p)
    return p


@pytest.fixture
def sentinel_product(prod_id, prod_name, sample_data_path, capsys):
    '''
    Returns uncompressed Sentinel product
    '''
    prod_path = os.path.join(sample_data_path, prod_name)
    if os.path.exists(prod_path):
        return prod_path
    
    download_dir = os.path.join(sample_data_path, prod_id)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    api = SentinelAPI(**credentials.scihub)
    with capsys.disabled():
        print('Downloading', prod_id, 'under', prod_name)
        api.download(prod_id, directory_path=download_dir)
    
    zipf = glob(os.path.join(download_dir, '*'))[0]
    
    with zipfile.ZipFile(zipf, mode='r') as z, capsys.disabled():
        print('Uncompressing', zipf)
        z.extractall(path=sample_data_path)
    
    assert os.path.exists(prod_path)
    
    return prod_path