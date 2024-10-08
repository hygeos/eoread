#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Define and download test products defined in products.py
"""

from pathlib import Path
from .download_legacy import download_multi
from core import config


def product_getter(request):
    """
    A function designed to be wrapped as a fixture for downloading products

    Example:
        product = pytest.fixture(params=['product1'])(product_getter)

    Or:
        product = pytest.fixture(product_getter)
    Then parametrized by:
        @pytest.mark.parametrize('product', ['product1', 'product2'], indirect=True)
    """
    prod = get_sample_products()[request.param]
    download_multi(prod)
    return prod


def get_sample_products(dir_samples=None):
    """
    definition of test products
    - name, product: path to the product
    - scihub_id, coda_id: key for downloading on scihub/coda
    - url: url for direct download
    - archive: basename of the downloaded file (defaults to the basename of 'url')
    """
    # TODO: don't use this function in the tests. Use a fixture based on product_getter instead
    if dir_samples is None:
        dir_samples = config.get('dir_samples')
    else:
        dir_samples = Path(dir_samples)

    products = {
        # Sentinel-2 MSI
        'prod_S2_L1_20190419': {
            'path': dir_samples/'MSI'/'S2A_MSIL1C_20190419T105621_N0207_R094_T31UDS_20190419T130656.SAFE',
            'scihub_id': '3ac99e56-8bff-4af3-b9bd-2e41a1ddaf61',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/MSI/S2A_MSIL1C_20190419T105621_N0207_R094_T31UDS_20190419T130656.zip',
        },

        # Sentinel-2 MSI, PB 4.00
        'prod_S2_L1_20220202': {
            'path': dir_samples/'MSI'/'S2A_MSIL1C_20220202T110251_N0400_R094_T31UDS_20220202T130715.SAFE',
        },

        # Sentinel-3 OLCI
        'prod_S3_L1_20190430': {
            'path': dir_samples/'OLCI'/'S3A_OL_1_EFR____20190430T094655_20190430T094955_20190501T131540_0179_044_136_2160_LN1_O_NT_002.SEN3',
            'scihub_id': '6271ae12-0e00-47d1-9a08-b0658d2262ad',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/OLCI/S3A_OL_1_EFR____20190430T094655_20190430T094955_20190501T131540_0179_044_136_2160_LN1_O_NT_002.zip',
            'ROI': {'sline': 3100, 'eline': 3300, 'scol': 3000, 'ecol': 3300}, # Corsica
        },
        'prod_S3_L2_20190612': {
            'path': dir_samples/'OLCI'/'S3B_OL_2_WFR____20190612T085520_20190612T085820_20190613T175523_0179_026_221_2340_MAR_O_NT_002.SEN3',
            'coda_id': 'a8c346c8-7752-4720-82b8-e5dea5fccf22',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/OLCI/S3B_OL_2_WFR____20190612T085520_20190612T085820_20190613T175523_0179_026_221_2340_MAR_O_NT_002.zip',
        },

        # MERIS
        'prod_meris_L1_20060822': {
            # from https://earth.esa.int/web/guest/-/meris-sample-data-4320
            'path': dir_samples/'MERIS'/'MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1',
            'url': 'https://earth.esa.int/c/document_library/get_file?folderId=23684&name=DLFE-451.zip',
        },
        'prod_meris_L1_20080701': {
            'path': dir_samples/'MERIS'/'MER_RR__1PRACR20080701_014028_000026402070_00003_33123_0000.N1',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/MERIS/MER_RR__1PRACR20080701_014028_000026402070_00003_33123_0000.N1.gz',
        },

        # MODIS Aqua
        'prod_A2008106_L1A_LAC': {
            # https://oceandata.sci.gsfc.nasa.gov/ob/getfile/A2008106124500.L1A_LAC.bz2
            'path': dir_samples/'MODIS'/'A2008106124500.L1A_LAC',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/MODIS/A2008106124500.L1A_LAC.bz2',
        },

        # MODIS Terra
        'prod_MODIST_L1A_2010100_L1A_LAC': {
            # https://oceandata.sci.gsfc.nasa.gov/ob/getfile/T2010100112000.L1A_LAC.bz2
            'path': dir_samples/'MODIS'/'T2010100112000.L1A_LAC',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/MODIS/T2010100112000.L1A_LAC.bz2',
        },

        # MODIS binned
        'prod_MODIS_binned_chl': {
            # https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20142732014280.L3b_8D_CHL.nc
            'path': dir_samples/'MODIS'/'A20142732014280.L3b_8D_CHL.nc',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/MODIS/A20142732014280.L3b_8D_CHL.nc',
        },

        # VIIRS
        'prod_V2019086_L1A_SNPP': {
            # https://oceandata.sci.gsfc.nasa.gov/ob/getfile/V2019086125400.L1A_SNPP.nc
            'path': dir_samples/'VIIRS'/'V2019086125400.L1A_SNPP.nc',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/VIIRS/V2019086125400.L1A_SNPP.nc',
        },

        # SeaWiFS
        'prod_S2004115_L1A_GAC': {
            # https://oceandata.sci.gsfc.nasa.gov/ob/getfile/S2004115125135.L1A_GAC.Z
            'path': dir_samples/'SeaWiFS'/'S2004115125135.L1A_GAC',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/SeaWiFS/S2004115125135.L1A_GAC.Z',
        },

        # Landsat-8 OLI
        'prod_oli_L1': {
            'path': dir_samples/'LANDSAT8_OLI'/'LC80140282017275LGN00',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/LANDSAT8_OLI/LC80140282017275LGN00.tar.gz',
        },
        
        # GCOM-C1 SGLI
        'prod_sgli': {
            'path': dir_samples/'SGLI'/'GC1SG1_201912050159F05712_1BSG_VNRDQ_1007.h5',
            'url': 'http://download.hygeos.com/EOREAD_TESTDATA/SGLI/'
                'GC1SG1_201912050159F05712_1BSG_VNRDQ_1007.h5',
        },
    }

    return products
