from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from eoread.download import Mirror_Uncompress, download_S2_google
from ftplib import FTP
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from eoread import download
from eoread.nasa import nasa_download, nasa_download_uncompress
from eoread.uncompress import uncompress
from fs.ftpfs import FTPFS

@pytest.mark.parametrize('product_name', [
    'S2B_MSIL1C_20201217T111359_N0209_R137_T30TWT_20201217T132006',
    'S2B_MSIL1C_20180708T092029_N0206_R093_T33NVE_20180708T132508',
    # 'S2B_MSIL2A_20190901T105619_N0213_R094_T30TWT_20190901T141237',    # Not implemented
])
def test_download_S2_google(product_name):
    with TemporaryDirectory() as tmpdir:
        f = download_S2_google(product_name, tmpdir)
        assert f.exists()
    

@pytest.mark.parametrize('product_name', [
    'S3A_OL_1_EFR____20220320T221328_20220320T221334_20220322T021726_0006_083_172_1440_LN1_O_NT_002',   # small sample product
    'S2000001002712.L1A_GAC.Z',
])
@pytest.mark.parametrize('do_uncompress', [True, False])
def test_download_nasa(product_name, do_uncompress):
    '''
    Test downloading (and uncompressing) some files
    '''
    with TemporaryDirectory() as tmpdir:
        if do_uncompress:
            for _ in range(2):
                f = nasa_download_uncompress(product_name, tmpdir)
                assert f.exists()
        else:
            f = nasa_download(product_name, Path(tmpdir)/'compressed')
            assert f.exists()
            assert uncompress(f, Path(tmpdir)/'uncompressed').exists()


def test_download_missing():
    '''
    Behaviour in case of missing file
    '''
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            nasa_download('ABCDEFG0123456789', tmpdir)


@pytest.mark.parametrize('ftp_object', [
    # FTPFS('ftp.us.debian.org').opendir('debian/dists/Debian8.11/main/installer-armhf/current/images/hd-media/'),
    'ftp://ftp.us.debian.org/debian/dists/Debian8.11/main/installer-armhf/current/images/hd-media/',
    ])
def test_mirror_uncompress(ftp_object):

    with TemporaryDirectory() as tmpdir:
        mfs = Mirror_Uncompress(
            ftp_object,
            tmpdir)
        print(list(mfs.glob('*')))

        mfs.get('boot.scr')
        mfs.get('boot.scr')
        mfs.get('hd-media.tar.gz')
        mfs.get('hd-media')
        mfs.get('SD-card-images/partition.img.gz')
        mfs.get('SD-card-images/partition.img')
        mfs.get('SD-card-images')
        mfs.get('SD-card-images')
        mfs.get('initrd.gz')
        mfs.get('initrd')

        mfs.get_local().tree()

        # test get without ftp
        mfs2 = Mirror_Uncompress(
            None,
            tmpdir)
        mfs2.get('boot.scr')


def test_ftp():
    ftp = FTP('ftp.de.debian.org')
    ftp.login()
    ls = download.ftp_list(ftp, '/debian/doc/')
    assert ls
    with TemporaryDirectory() as tmpdir:
        download.ftp_download(ftp, Path(tmpdir)/'00-INDEX', '/debian/doc/')
    

