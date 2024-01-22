import pytest

from os.path import exists
from tempfile import TemporaryDirectory
from eoread.download import DownloadSatellite


@pytest.mark.parametrize('product',[
    ""
])
def test_L5(product):
    aoi   = -4.15, -3.58, 46.16, 46.51
    start = '2000-05-15'
    end   = '2002-08-01'
    sat_collec = "LANDSAT-5"

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
    ""
])
def test_L7(product):
    aoi   = -4.15, -3.58, 46.16, 46.51
    start = '2020-05-15'
    end   = '2022-08-01'
    sat_collec = "LANDSAT-7"

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
    ""
])
def test_L8(product):
    aoi   = -4.15, -3.58, 46.16, 46.51
    start = '2020-05-15'
    end   = '2022-08-01'
    sat_collec = "LANDSAT-8"

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
    ""
])
def test_L9(product):
    aoi   = -4.15, -3.58, 46.16, 46.51
    start = '2020-05-15'
    end   = '2022-08-01'
    sat_collec = "LANDSAT-9"

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