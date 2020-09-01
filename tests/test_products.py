#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Define and download test products defined in products.py
"""

import pytest
from eoread.download import download
from eoread.sample_products import get_sample_products

products = get_sample_products()

@pytest.mark.parametrize('product', products.values(),
                         ids=list(products.keys()))
def test_available(product):
    path = product['path']
    if not path.exists():
        raise Exception(
            f'{path} is missing. '
            'You may run `python -m tests.test_products` to download.')


if __name__ == "__main__":
    print('Downloading sample products...')
    for _, p in products.items():
        download(p)
