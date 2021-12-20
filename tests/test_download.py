import pytest
from tempfile import TemporaryDirectory
from eoread.download import get_S2_google_url

@pytest.mark.parametrize('product_name', [
    'S2B_MSIL1C_20201217T111359_N0209_R137_T30TWT_20201217T132006',
    'S2B_MSIL1C_20180708T092029_N0206_R093_T33NVE_20180708T132508',
    'S2B_MSIL2A_20190901T105619_N0213_R094_T30TWT_20190901T141237',
])
def test_download_S2_google(product_name):
    from fels import fels
    with TemporaryDirectory() as tmpdir:
        url = get_S2_google_url(product_name)
        fels.get_sentinel2_image(url, tmpdir)
