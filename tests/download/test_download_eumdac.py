import pytest

from os.path import exists
from tempfile import TemporaryDirectory
from src.download import DownloadSatellite
from src.download_eumdac import DownloadEumetsat


@pytest.mark.parametrize('product',[
    "",
])
def test_SEVIRI(product):
    aoi   = 46.16, 46.51, -16.15, -15.58
    start = '2021-11-01'
    end   = '2021-12-01'
    sat_collec = "EUMET-SEVIRI"

    with TemporaryDirectory() as tmpdir:
        dl = DownloadEumetsat(data_collection = sat_collec,
                              bbox = aoi,
                              start_date = start, 
                              end_date = end,
                              save_dir = tmpdir,
                              product = product)        
        filepath = dl.get(list_id=dl.product[:1], 
                               zip_format=True)[0]
        assert exists(filepath)