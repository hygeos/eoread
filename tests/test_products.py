#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Define and download test products defined in products.py
"""

import pytest
from eoread.download import download
from tests.products import products, get_path, dir_samples

@pytest.mark.parametrize('product', products.values(),
                         ids=list(products.keys()))
def test_available(product):
    path = get_path(product)
    if not path.exists():
        raise Exception(
            f'{path} is missing. '
            'You may run `python -m tests.test_products` to download.')


if __name__ == "__main__":
    print('Downloading sample products...')
    for k, v in products.items():
        download(v, dir_samples)
