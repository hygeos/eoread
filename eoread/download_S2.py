import os
import shutil
import glob
from tempfile import NamedTemporaryFile
from urllib.request import urlopen, HTTPError

import requests

"""
Copied from fels and adapted to work with L2A files
https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud
"""


def get_sentinel2_image(url, outputdir, overwrite=False,
                        reject_old=False):
    """
    Collect the entire dir structure of the image files from the
    manifest.safe file and build the same structure in the output
    location.

    Returns:
        True if image was downloaded
        False if partial=False and image was not fully downloaded
            or if reject_old=True and it is old-format
            or if noinspire=False and INSPIRE file is missing
    """
    img = os.path.basename(url)
    target_path = os.path.join(outputdir, img)
    target_manifest = os.path.join(target_path, "manifest.safe")

    return_status = True
    if not os.path.exists(target_path) or overwrite:

        manifest_url = url + "/manifest.safe"

        if reject_old:
            # check contents of manifest before downloading the rest
            content = urlopen(manifest_url)
            with NamedTemporaryFile() as f:
                shutil.copyfileobj(content, f)
                if not is_new(f.name):
                    return False

        os.makedirs(target_path, exist_ok=True)
        content = urlopen(manifest_url)
        with open(target_manifest, 'wb') as f:
            shutil.copyfileobj(content, f)
        with open(target_manifest, 'r') as manifest_file:
            manifest_lines = manifest_file.read().split()
        for line in manifest_lines:
            if 'href' in line:
                rel_path = line[line.find('href=".')+7:]
                rel_path = rel_path[:rel_path.find('"')]
                abs_path = os.path.join(target_path, *rel_path.split('/')[1:])
                if not os.path.exists(os.path.dirname(abs_path)):
                    os.makedirs(os.path.dirname(abs_path))
                try:
                    download_file(url + rel_path, abs_path)
                except HTTPError as error:
                    print("Error downloading {} [{}]".format(url + rel_path, error))
                    continue
                if not abs_path.endswith('.html'):
                    with open(abs_path) as fp:
                        file_start = fp.read(6)
                        assert file_start != '<html>', \
                            f'Fetched file was html ({abs_path})'
                    
    elif reject_old and not is_new(target_manifest):
        print(f'Warning: old-format image {outputdir} exists')
        return_status = False

    return return_status


def is_new(safedir_or_manifest):
    '''
    Check if a S2 scene is in the new (after Nov 2016) format.

    If the scene is already downloaded, the safedir directory structure can be crawled to determine this.
    If not, download the manifest.safe first for an equivalent check.

    Example:
        >>> safedir = 'S2A_MSIL1C_20160106T021717_N0201_R103_T52SDG_20160106T094733.SAFE/'
        >>> manifest = os.path.join(safedir, 'manifest.safe')
        >>> assert is_new(safedir) == False
        >>> assert is_new(manifest) == False
    '''
    if os.path.isdir(safedir_or_manifest):
        safedir = safedir_or_manifest
        # if this file does not have the standard name (len==0), the scene is old format.
        # if it is duplicated (len>1), there are multiple granuledirs and we don't want that.
        return len(glob.glob(os.path.join(safedir, 'GRANULE', '*', 'MTD_TL.xml'))) == 1

    elif os.path.isfile(safedir_or_manifest):
        manifest = safedir_or_manifest
        with open(manifest, 'r') as f:
            lines = f.read().split()
        return len([l for l in lines if 'MTD_TL.xml' in l]) == 1

    else:
        raise ValueError(f'{safedir_or_manifest} is not a safedir or manifest')


def download_file(url, destination_filename):
    """Function to download files using pycurl lib"""
    with requests.get(url, stream=True) as r:
        with open(destination_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
