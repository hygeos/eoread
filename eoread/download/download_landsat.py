import pandas as pd
import requests
import json
import os
import re
import time
import shutil

from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile 
from datetime import datetime, timedelta
from urllib.parse import urljoin
# from landsatxplore.earthexplorer import EarthExplorer, EarthExplorerError

from eoread.download.download_base import DownloadBase, UnauthorizedError


# BASED ON : https://github.com/yannforget/landsatxplore/tree/master/landsatxplore

class DownloadLandsat(DownloadBase):

    def __init__(self, 
                 data_collection: str, 
                 save_dir: str | Path, 
                 start_date: str | datetime, 
                 lon_min: float = None, 
                 lon_max: float = None, 
                 lat_min: float = None, 
                 lat_max: float = None, 
                 bbox: list[float] = None,
                 point: dict[float] = None,
                 inter_width: int | timedelta = None,
                 end_date: str | datetime = None,
                 product: str = None,
                 level: int = 1):
        
        super().__init__(data_collection, save_dir, start_date,
                         lon_min, lon_max, lat_min, lat_max,
                         bbox, point, inter_width, end_date,
                         product, level)
        
        self.max_cloud_cover = 100
        
        credentials = self._get_auth("cr.usgs.gov")
        self.client_id     = credentials['user']
        self.client_secret = credentials['password']
        self.api_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
        self.login(self.client_id,self.client_secret)
        self.request_available()

    def request_available(self):
        dataset    = self._get_dataset_name()
        
        # Configure scene constraints for request
        spatial_filter = {}
        spatial_filter["filterType"] = "mbr"
        if self.point:
            spatial_filter["lowerLeft"]  = {"latitude":self.point[0], 
                                            "longitude":self.point[1]}
            spatial_filter["upperRight"] = spatial_filter["lowerLeft"]

        elif self.bbox:
            spatial_filter["lowerLeft"]  = {"latitude":self.bbox[2], 
                                            "longitude":self.bbox[0]}
            spatial_filter["upperRight"] = {"latitude":self.bbox[3], 
                                            "longitude":self.bbox[1]}
        
        acquisition_filter = {"start": datetime.strftime(self.start, "%Y-%m-%d %H:%M:%S"),
                              "end"  : datetime.strftime(self.end, "%Y-%m-%d %H:%M:%S")}

        cloud_cover_filter = {"min" : 0,
                              "max" : self.max_cloud_cover,
                              "includeUnknown" : False}

        scene_filter = {"acquisitionFilter": acquisition_filter,
                        "spatialFilter"    : spatial_filter,
                        "cloudCoverFilter" : cloud_cover_filter,
                        "metadataFilter"   : None,
                        "seasonalFilter"   : None}

        params={
            "datasetName": dataset,
            "sceneFilter": scene_filter,
            "maxResults": 100,
            "metadataType": "full",
        }

        # Retrieves scenes corresponding to the constraints  
        url = urljoin(self.api_url, "scene-search")
        data = json.dumps(params)
        response = self.request_get(self.session_api, url, data=data).json()
        if response.get("errorCode"):
            raise RuntimeError(f"{response.get('errorCode')}: {response.get('errorMessage')}.")
        r = response.get("data")
        scenes = [scene for scene in r.get("results")]

        # Generates various outputs in dataframe format
        lp = pd.DataFrame(scenes)
        check_prod = lp.apply(lambda x: self.basename_prod in x['entityId'], axis=1)
        self.list_product = lp[check_prod]
        self.list_prod_id = list(self.list_product['displayId'])
        assert len(self.list_prod_id) != 0, 'No data found according to your request'
        self.print_msg(f'Finds products matching the query ({len(self.list_prod_id)} products)')

    def login(self, username: str, password: str):
        self.session_api = requests.Session()
        login_url = urljoin(self.api_url, "login")
        payload = {"username": username, "password": password}
        niter = 0
        response = self.session_api.post(login_url, json.dumps(payload)).json()
        while response.get("errorCode"):
            time.sleep(3)
            response = self.session_api.post(login_url, json.dumps(payload)).json()
            niter += 1
            if niter == 10:
                raise RuntimeError(f"{response.get('errorCode')}: {response.get('errorMessage')}.")
        self.session_api.headers["X-Auth-Token"] = response.get("data")
        self.print_msg(f'Log to API ({self.api_url})')

    def download_prod(self,
                      scene: str,
                      dataset: str):
        entity_id = str(self.list_product[self.list_product['displayId'] == scene]['entityId'].iloc[0])
        dataset_id_list = DATA_PRODUCTS[dataset]

        for i, dataset_id in enumerate(dataset_id_list):
            # Attempts to obtain a download url for each dataset  
            url =  f"https://earthexplorer.usgs.gov/download/{dataset_id}/{entity_id}/EE/"
            r = self.session.get(url, allow_redirects=False, stream=True, timeout=300)
            r.raise_for_status()
            error_msg = r.json().get("errorMessage")
            if error_msg:
                continue
            download_url = r.json().get("url")

            # get information about the file to be downloaded 
            response = self.request_get(self.session, download_url, stream=True, allow_redirects=True, timeout=300)
            filesize = int(response.headers.get("Content-Length"))
            pbar = tqdm(total=filesize, unit_scale=True, unit="B",
                        unit_divisor=1024, leave=False)
            pbar.set_description(f"Downloading {scene[:10]}")  
            local_filename = response.headers["Content-Disposition"].split("=")[-1]
            local_filename = local_filename.replace('"', "")
            local_filename = os.path.join(self.save_dir, local_filename)

            # Check if the file already exists 
            headers = {}
            file_mode = "wb"
            file_exists = os.path.exists(local_filename)
            if file_exists:
                downloaded_bytes = os.path.getsize(local_filename)
                headers = {"Range": f"bytes={downloaded_bytes}-"}
                file_mode = "ab"
                if downloaded_bytes == filesize:
                    return local_filename

            # Download file
            r = self.request_get(self.session, download_url, stream=True, 
                                 allow_redirects=True, headers=headers, timeout=300)
            with open(local_filename, file_mode) as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(1024)
            return local_filename

    def get(self, list_id: list = None, tar_format: bool = False):  
        # Initialize session for download
        self.print_msg('Start downloading')
        self.session = requests.Session()
        rsp = self.session.get("https://ers.cr.usgs.gov/login/")
        csrf = re.findall(r'name="csrf" value="(.+?)"', rsp.text)[0]
        payload = {"username": self.client_id,
                   "password": self.client_secret,
                   "csrf": csrf}
        rsp = self.session.post("https://ers.cr.usgs.gov/login/", data=payload, allow_redirects=True)
        if not bool(self.session.cookies.get("EROS_SSO_production_secure")):
            raise UnauthorizedError

        # Prepare the progress bar
        terminal_width, _ = shutil.get_terminal_size()
        if list_id:
            pbar = tqdm(list_id, ncols=terminal_width)
        else:
            pbar = tqdm(self.list_product['displayId'], ncols=terminal_width)
        pbar.set_description("Download Sentinel files")
        
        # Download each scene
        output_path = []
        dataset = self._get_dataset_name()
        os.makedirs(self.save_dir, exist_ok=True)
        for scene in pbar:
            product_name = self.download_prod(scene, dataset)

            # Uncompress file if required
            if tar_format:
                output_path.append(product_name)
            else:
                zip_file = ZipFile(product_name, 'r')
                zip_file.extractall(self.save_dir)
                os.remove(product_name)
                output_path.append(os.path.join(self.save_dir, product_name[:-4]))

        self.session.get("https://earthexplorer.usgs.gov/logout")
        self.print_msg(f'All {len(pbar)} files have been successfully downloaded')
        return output_path

    def _get_dataset_name(self):
        collec_name = ''
        if self.data_collection == 'LANDSAT-5':
            self.basename_prod = 'LT5'
            collec_name += 'landsat_tm_c2'
        if self.data_collection == 'LANDSAT-7':
            self.basename_prod = 'LE7'
            collec_name += 'landsat_etm_c2'
        if self.data_collection == 'LANDSAT-8':
            self.basename_prod = 'LC8'
            collec_name += 'landsat_ot_c2'
        if self.data_collection == 'LANDSAT-9':
            self.basename_prod = 'LC9'
            collec_name += 'landsat_ot_c2'
        
        if self.level == 1:
            collec_name += '_l1'
        if self.level == 2:
            collec_name += '_l2'
        
        return collec_name

DATA_PRODUCTS = {
    # Level 1 datasets
    "landsat_tm_c2_l1": ["5e81f14f92acf9ef", "5e83d0a0f94d7d8d", "63231219fdd8c4e5"],
    "landsat_etm_c2_l1":[ "5e83d0d0d2aaa488", "5e83d0d08fec8a66"],
    "landsat_ot_c2_l1": ["632211e26883b1f7", "5e81f14ff4f9941c", "5e81f14f92acf9ef"],
    # Level 2 datasets
    "landsat_tm_c2_l2": ["5e83d11933473426", "5e83d11933473426", "632312ba6c0988ef"],
    "landsat_etm_c2_l2": ["5e83d12aada2e3c5", "5e83d12aed0efa58", "632311068b0935a8"],
    "landsat_ot_c2_l2": ["5e83d14f30ea90a9", "5e83d14fec7cae84", "632210d4770592cf"]
}