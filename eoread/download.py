#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities to download products
"""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from sentinelsat import SentinelAPI
import subprocess
from netrc import netrc
import threading
from typing import Union

from tqdm import tqdm

from eoread.download_S2 import get_sentinel2_image
from eoread.uncompress import uncompress as uncomp, uncompress_decorator
from ftplib import FTP, error_perm
import fnmatch
from .common import timeit
from .utils.fileutils import filegen


def download_url(url, dirname, wget_opts='',
                 check_function=None,
                 verbose=True,
                 **kwargs
                 ):
    """
    Download `url` to `dirname` with wget

    Options `wget_opts` are added to wget
    Uses a `filegen` wrapper
    Other kwargs are passed to `filegen` (lock_timeout, tmpdir, if_exists)

    Returns the path to the downloaded file
    """
    target = Path(dirname)/(Path(url).name)
    if verbose:
        print('Downloading:', url)
        print('To: ', target)
    
    @filegen(**kwargs)
    def download_target(path):
        cmd = f'wget {wget_opts} {url} -O {path}'
        # Detect access problem
        if subprocess.call(cmd.split()):
            raise RuntimeError(f'Authentification issue : "{cmd}"')

        if check_function is not None:
            check_function(path)

    download_target(target)

    return target


def get_auth(name):
    """
    Returns a dictionary with credentials, using .netrc

    `name` is the identifier (= `machine` in .netrc). This allows for several accounts
    on a single machine.
    The url is returned as `account`
    """
    ret = netrc().authenticators(name)
    if ret is None:
        raise ValueError(
            f'Please provide entry "{name}" in ~/.netrc ; '
            f'example: machine {name} login <login> password <passwd> account <url>')
    (login, account, password) = ret

    return {'user': login,
            'password': password,
            'url': account}


def get_auth_dhus(name):
    auth = get_auth(name)
    api_url = auth['url'] or {
        'scihub': 'https://scihub.copernicus.eu/dhus/',
        'coda': 'https://coda.eumetsat.int',
    }[name]
    return {'user': auth['user'],
            'password': auth['password'],
            'api_url': api_url}


def get_auth_ftp(name):
    """
    get netrc credentials for use with pyfilesystem's
    FTPFS or ftplib's FTP
    
    Ex: FTP(**get_auth_ftp(<name>))
    """
    auth = get_auth(name)
    return {'host': auth['url'],
            'user': auth['user'],
            'passwd': auth['password']}

def get_url_ftpfs(name):
    """
    get netrc credentials for use with pyfilesystem's fs.open_fs

    Ex: fs.open_fs(get_url_ftpfs(<name>))
    """
    auth = get_auth(name)
    user = auth['user']
    password = auth['password']
    machine = auth['url']
    return f"ftp://{user}:{password}@{machine}/"


def download_sentinel(product, dirname):
    """
    Download a sentinel product to `dirname`
    """
    if 'scihub_id' in product:
        cred = get_auth('scihub')
        pid = product['scihub_id']
    elif 'coda_id' in product:
        cred = get_auth('coda')
        pid = product['coda_id']
    else:
        return None

    api = SentinelAPI(**cred)
    api.download(pid, directory_path=dirname)


@filegen()
def download_sentinelapi(target: Path,
                         source: str = 'scihub'):
    """
    Download a product using sentinelapi

    Source: scihub, coda
    """
    api = SentinelAPI(**get_auth_dhus(source))
    res = list(api.query(filename=target.name+'*'))
    assert len(res) == 1
    with TemporaryDirectory() as tmpdir:
        compressed = api.download(
            res[0],
            directory_path=tmpdir)
        uncompressed = uncomp(compressed['path'], tmpdir)
        shutil.move(uncompressed, target)


def download_multi(product):
    """
    Download a product from various sources
        - direct url
        - sentinel hubs

    Arguments:
    ----------

    product: dict with following keys:
        - 'path' local target path (pathlib object)
          Example:
            - Path()/'S2A_MSIL1C_2019[...].SAFE',
            - 'S3A_OL_1_EFR____2019[...].SEN3'
        Download source (one or several)
        - 'scihub_id': '6271ae12-0e00-47d1-9a08-b0658d2262ad',
        - 'coda_id': 'a8c346c8-7752-4720-82b8-e5dea5fccf22',
        - 'url': 'https://earth.esa.int/[...]DLFE-451.zip',
    
    Uncompresses the downloaded product if necessary

    Returns: the path to the product
    """
    path = product['path']
    print(f'Getting {path}')

    if path.exists():
        print('Skipping existing product', path)
        return path

    with TemporaryDirectory(prefix='tmp_eoread_download_') as tmpdir:
        if 'url' in product:
            download_url(product['url'], tmpdir)
        elif product['path'].name.startswith('S2'):
            import fels
            url = get_S2_google_url(product['path'].name)
            fels.get_sentinel2_image(url, tmpdir)
        elif ('scihub_id' in product) or ('coda_id' in product):
            download_sentinel(product, tmpdir)
        else:
            raise Exception(
                f'No valid method for retrieving product {path.name}')

        compressed = next(Path(tmpdir).glob('*'))

        uncomp(compressed, path.parent, on_uncompressed='copy')

    assert path.exists(), f'{path} does not exist.'

    return path


def get_S2_google_url(filename):
    """
    Get the google url for a given S2 product, like
    'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/32/T/QR/S2A_MSIL1C_20170111T100351_N0204_R122_T32TQR_20170111T100351.SAFE'

    The result of this function can be downloaded with the fels module:
        fels.get_sentinel2_image(url, directory)
    Note: the filename can be obtained either with the Sentinels hub api, or with
    google's catalog (see fels)
    """
    tile = filename.split('_')[-2]
    prod_type = filename.split('_')[1]
    assert len(tile) == 6
    utm = tile[1:3]
    pos0 = tile[3::4]
    pos1 = tile[4:]
    if prod_type == 'MSIL1C':
        url_base = 'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles'
    elif prod_type == 'MSIL2A':
        url_base = 'http://storage.googleapis.com/gcp-public-data-sentinel-2/L2/tiles'
        # gs://gcp-public-data-sentinel-2/L2/tiles/32/T/NS/S2B_MSIL2A_20210511T101559_N0300_R065_T32TNS_20210511T134528.SAFE/
    else:
        raise Exception(f'Unexpected product type "{prod_type}"')
    filename_full = filename if (filename.endswith('.SAFE')) else (filename+'.SAFE')
    url = f'{url_base}/{utm}/{pos0}/{pos1}/{filename_full}'

    return url

def download_S2_google(product, dirname, **kwargs) -> Path:
    target = Path(dirname)/(product+'.SAFE')
    @filegen(**kwargs)
    def down_S2(path):
        with timeit('Downloading'):
            print(f'Downloading {product}...')
            url = get_S2_google_url(product)
            get_sentinel2_image(url, outputdir=path.parent)
    down_S2(target)
    return target


@filegen(1)
def ftp_download(ftp: FTP, file_local: Path, dir_server: str, verbose=True):
    """
    Downloads `file_local` on ftp, from server directory `dir_server`

    The file name on the server is determined by `file_local.name`

    Refs:
        https://stackoverflow.com/questions/19692739/
        https://stackoverflow.com/questions/73534659/
    """
    fname = file_local.name
    path_server = str(Path('/')/str(dir_server)/fname)
    size = ftp.size(path_server)
    pbar = tqdm(desc=f'Downloading {path_server}',
                total=size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                disable=not verbose)
    with open(file_local, 'wb') as fp, ftp.transfercmd('RETR ' + path_server) as sock:
        def background():
            while True:
                block = sock.recv(1024*1024)
                if not block:
                    break
                fp.write(block)
                pbar.update(len(block))
        t = threading.Thread(target=background)
        t.start()
        noops_sent = 0
        while t.is_alive():
            t.join(60)
            ftp.putcmd('NOOP')
            noops_sent += 1
    pbar.close()
    ftp.voidresp()
    for _ in range(noops_sent):
        ftp.voidresp()
    assert file_local.exists()


def ftp_file_exists(ftp: FTP, path_server: Union[Path, str]) -> bool:
    try:  # test file existence
        ftp.size(str(path_server))
        return True
    except error_perm:
        return False

def ftp_create_dir(ftp: FTP, path_server: Union[Path, str]):
    try:
        ftp.cwd(str(path_server))
    except error_perm:
        # if path_server does not exist
        assert Path(path_server).parent != Path(path_server)
        ftp_create_dir(ftp, Path(path_server).parent)
        ftp.mkd(str(path_server))
    ftp.cwd('/')


def ftp_upload(ftp: FTP,
               file_local: Path,
               dir_server: str,
               if_exists='skip',
               blocksize=8192,
               verbose=True,
               ):
    """
    FTP upload function
    
    - Use temporary files
    - Create remote directories
    - if_exists:
        'skip': skip the existing file
        'error': raise an error on existing file
        'overwrite': overwrite existing file
    """
    path_server = f'{dir_server}/{file_local.name}'
    if ftp_file_exists(ftp, path_server):
        if if_exists == 'skip':
            # check size consistency
            assert file_local.stat().st_size == ftp.size(path_server)
            return
        elif if_exists == 'overwrite':
            ftp.delete(path_server)
        elif if_exists == 'error':
            raise FileExistsError
        else:
            raise ValueError(f'skip = {if_exists}')

    assert not ftp_file_exists(ftp, path_server)
    
    # cleanup tmp file
    path_server_tmp = f'{dir_server}/{file_local.name}'
    if ftp_file_exists(ftp, path_server_tmp):
        ftp.delete(path_server_tmp)

    # create directory
    ftp_create_dir(ftp, dir_server)
        
    cmd = f'STOR {path_server_tmp}'
    pbar = tqdm(desc=f'Uploading {path_server}',
                total=file_local.stat().st_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                disable=not verbose)
    with ftp.transfercmd(cmd) as sock, open(file_local, 'rb') as fp:
        def background():
            while True:
                buf = fp.read(blocksize)
                if not buf:
                    break
                sock.sendall(buf)
                pbar.update(len(buf))
        t = threading.Thread(target=background)
        t.start()
        noops_sent = 0
        while t.is_alive():
            t.join(60)
            ftp.putcmd('NOOP')
            noops_sent += 1
    pbar.close()
    ftp.voidresp()
    for _ in range(noops_sent):
        ftp.voidresp()
    ftp.rename(path_server_tmp, path_server)


def ftp_list(ftp: FTP, dir_server: str, pattern: str='*'):
    '''
    Returns the list of fles matching `pattern` on `dir_server`
    '''
    ftp.cwd(dir_server)
    ls = ftp.nlst()
    return fnmatch.filter(ls, pattern)
