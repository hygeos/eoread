import pytest
from tempfile import TemporaryDirectory
from eoread.mirror import Mirror_Uncompress


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