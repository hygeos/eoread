import filecmp
import os
import random
import string
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from eoread.download_legacy import download_S2_google
from eoread import download
from eoread.nasa import nasa_download, nasa_download_uncompress
from eoread.utils.uncompress import uncompress
from ftplib import FTP

@pytest.mark.parametrize('product_name', [
    'S2B_MSIL1C_20201217T111359_N0209_R137_T30TWT_20201217T132006',
    'S2B_MSIL1C_20180708T092029_N0206_R093_T33NVE_20180708T132508',
    'S2B_MSIL2A_20190901T105619_N0213_R094_T30TWT_20190901T141237',
])
def test_download_S2_google(product_name):
    with TemporaryDirectory() as tmpdir:
        f = download_S2_google(product_name, tmpdir)
        assert f.exists()
    

@pytest.mark.parametrize('product_name', [
    # 'S3A_OL_1_EFR____20220320T221328_20220320T221334_20220322T021726_0006_083_172_1440_LN1_O_NT_002',   # small sample product
    'S2000001002712.L1A_GAC.Z',
    'GMAO_MERRA2.20230119T140000.MET.nc',
])
def test_download_nasa(product_name):
    '''
    Test downloading (and uncompressing) some files
    '''
    with TemporaryDirectory() as tmpdir:
        f = nasa_download(product_name, tmpdir)
        assert f.exists()


def test_download_missing():
    '''
    Behaviour in case of missing file
    '''
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            nasa_download('ABCDEFG0123456789', tmpdir)


def test_auth_error():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError):
            nasa_download('GMAO_MERRA2.20230119T140000.MET.nc', tmpdir,
                          wget_extra='--no-netrc')


def test_ftp_download():
    ftp = FTP('test.rebex.net',
              user='demo',
              passwd='password')
    ls = download.ftp_list(ftp, '/')
    assert ls
    with TemporaryDirectory() as tmpdir:
        download.ftp_download(ftp, Path(tmpdir)/'readme.txt', '/pub/example/')
        

@pytest.mark.parametrize('size_bytes', [
    1024,
    100000,
])
def test_ftp_upload(size_bytes):
    # Connect to some public test ftp server (with write permissions)
    ftp = FTP('ftp.dlptest.com',
              user='dlpuser',
              passwd='rNrKYTX9g7z3RgJRmxWuGHbeu')
    rands = ''.join(random.choice(string.ascii_letters)
                       for _ in range(16))
    dir_server = '/ftp_upload_test_'+rands
    with TemporaryDirectory() as tmpdir:
        for _ in range(2): # twice, to check overwrite
            # create a random file
            tmpfile = Path(tmpdir)/rands
            if tmpfile.exists():
                tmpfile.unlink()
            with open(tmpfile, 'wb') as fout:
                fout.write(os.urandom(size_bytes))

            for if_exists in ['overwrite', 'skip']:
                # second upload should do nothing
                download.ftp_upload(ftp, tmpfile, dir_server, if_exists=if_exists)

            with pytest.raises(FileExistsError):
                download.ftp_upload(ftp, tmpfile, dir_server, if_exists='error')
            
            # Check consistency
            tmpfile2 = Path(tmpdir)/'check'/rands
            if tmpfile2.exists():
                tmpfile2.unlink()
            download.ftp_download(ftp, tmpfile2, dir_server)

            assert filecmp.cmp(tmpfile, tmpfile2)

        ftp.delete(str(Path(dir_server, rands)))
