#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Define and download test products defined in products.py
"""

from pathlib import Path

dir_base = Path(__file__).resolve().parent.parent
dir_samples = dir_base/'SAMPLE_DATA'

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
        'ROI': {'sline': 3100, 'eline': 3300, 'scol': 3000, 'ecol': 3300}, # Corsica
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
    return get_dir(product)/product['name']

def get_dir(product):
    """ Returns the directory containing the product """
    return dir_samples/product['folder']

