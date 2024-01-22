import pytest

from os.path import exists
from tempfile import TemporaryDirectory
from eoread.download import DownloadSatellite


@pytest.mark.skip()
@pytest.mark.parametrize('product',[
    "RAW",
    "SLC"
])
def test_S1(product):
    aoi   = 46.16, 46.51, -16.15, -15.58
    start = '2021-07-15'
    end   = '2021-08-01'
    sat_collec = "SENTINEL-1"

    with TemporaryDirectory() as tmpdir:
        dl = DownloadSatellite(data_collection = sat_collec,
                              bbox = aoi,
                              start_date = start, 
                              end_date = end,
                              save_dir = tmpdir,
                              product = product)
        filepath = dl.download(list_id=dl.api.list_prod_id[:1], 
                               compress_format=True)[0]
        assert exists(filepath)

@pytest.mark.parametrize('product',[
    "S2B_MSIL1C",
    "S2A_MSIL1C",
])
def test_S2(product):
    aoi   = 46.16, 46.51, -16.15, -15.58
    start = '2021-07-15'
    end   = '2021-08-01'
    sat_collec = "SENTINEL-2"
    
    with TemporaryDirectory() as tmpdir:
        dl = DownloadSatellite(data_collection = sat_collec,
                              bbox = aoi,
                              start_date = start, 
                              end_date = end,
                              save_dir = tmpdir,
                              product = product)
        filepath = dl.download(list_id=dl.api.list_prod_id[:1], 
                               compress_format=True)[0]
        assert exists(filepath)

@pytest.mark.parametrize('product',[
    "SYNERGY",
    "OLCI"
])
def test_S3(product):
    aoi   = 46.16, 46.51, -16.15, -15.58
    start = '2021-08-18'
    end   = '2021-08-20'
    sat_collec = "SENTINEL-3"

    with TemporaryDirectory() as tmpdir:
        dl = DownloadSatellite(data_collection = sat_collec,
                              bbox = aoi,
                              start_date = start, 
                              end_date = end,
                              save_dir = tmpdir,
                              product = product)
        filepath = dl.download(list_id=dl.api.list_prod_id[:1], 
                               compress_format=True)[0]
        assert exists(filepath)

@pytest.mark.parametrize('product',[
    "L1B"
])
def test_S5(product):
    aoi   = 46.16, 46.51, -16.15, -15.58
    start = '2021-07-18'
    end   = '2021-08-20'
    sat_collec = "SENTINEL-5P"

    with TemporaryDirectory() as tmpdir:
        dl = DownloadSatellite(data_collection = sat_collec,
                              bbox = aoi,
                              start_date = start, 
                              end_date = end,
                              save_dir = tmpdir,
                              product = product)
        filepath = dl.download(list_id=dl.api.list_prod_id[:1], 
                               compress_format=True)[0]
        assert exists(filepath)