import pytest

from os.path import exists
from tempfile import TemporaryDirectory
from eoread.download import DownloadSatellite


@pytest.mark.parametrize('product',[
    ""
])
def test_M(product):
    aoi   = -4.15, -3.58, 46.16, 46.51
    start = '2020-05-15'
    end   = '2022-08-01'
    sat_collec = "NASA-MODIS"

    with TemporaryDirectory() as tmpdir:
        dl = DownloadSatellite(data_collection = sat_collec,
                              bbox = aoi,
                              start_date = start, 
                              end_date = end,
                              save_dir = tmpdir,
                              product = product)
        pass