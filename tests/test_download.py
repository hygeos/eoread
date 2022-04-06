from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from eoread.download import download_url, get_S2_google_url
from eoread.nasa import nasa_download
from eoread.uncompress import uncompress

@pytest.mark.parametrize('product_name', [
    'S2B_MSIL1C_20201217T111359_N0209_R137_T30TWT_20201217T132006',
    'S2B_MSIL1C_20180708T092029_N0206_R093_T33NVE_20180708T132508',
    # 'S2B_MSIL2A_20190901T105619_N0213_R094_T30TWT_20190901T141237',    # Not implemented
])
def test_download_S2_google(product_name):
    from fels import fels
    with TemporaryDirectory() as tmpdir:
        url = get_S2_google_url(product_name)
        fels.get_sentinel2_image(url, tmpdir)

@pytest.mark.parametrize('product_name', [
    'S3A_OL_1_EFR____20190107T005248_20190107T005307_20190108T121012_0019_040_059_4680_LN1_O_NT_002',
    'S2000001002712.L1A_GAC.Z',
])
def test_download_nasa(product_name):
    '''
    Test downloading (and uncompressing) some files
    '''
    with TemporaryDirectory() as tmpdir:
        f = nasa_download(product_name, Path(tmpdir)/'compressed')
        print(f)
        assert f.exists()
        assert uncompress(f, Path(tmpdir)/'uncompressed').exists()


def test_download_missing():
    '''
    Behaviour in case of missing file
    '''
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            nasa_download('ABCDEFG0123456789', tmpdir)

