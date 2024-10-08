from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from eoread.download_eumdac import download_product, query, download_eumdac


def test_query():
    query(
        collection='EO:EUM:DAT:MSG:HRSEVIRI',
        title='MSG4-SEVI-MSG15-0100-NA-20221110081242.653000000Z-NA',
    )


@pytest.mark.parametrize('product', [
    'MSG4-SEVI-MSG15-0100-NA-20221110081242.653000000Z-NA',
    'S3B_OL_1_ERR____20230403T063939_20230403T064139_20230403T090437_0120_078_034______MAR_O_NR_002.SEN3',
])
def test_download(product):
    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)/product
        download_eumdac(target)
