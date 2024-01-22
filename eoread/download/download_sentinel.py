import requests
import shutil
import os
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile 
from datetime import datetime, timedelta
from shapely.geometry import Polygon

from eoread.download.download_base import DownloadBase


class DownloadSentinel(DownloadBase):

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
        
        assert hasattr(self,'bbox')

        self.start = datetime.strftime(self.start, self.date_format)
        self.end   = datetime.strftime(self.end, self.date_format)

        credentials = self._get_auth("dataspace.copernicus.eu")
        self.client_id     = credentials['user']
        self.client_secret = credentials['password']
        self.login(self.client_id,self.client_secret)
        self.request_available()

    def request_available(self):
        # Configure scene constraints for request
        poly = Polygon(((self.bbox[0],self.bbox[2]),(self.bbox[1],self.bbox[2]),
                        (self.bbox[1],self.bbox[3]),(self.bbox[0],self.bbox[3])))
        aoi = str(poly)+"'"
        json = requests.get(f"""https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name 
                            eq '{self.data_collection}' and 
                            OData.CSC.Intersects(area=geography'SRID=4326;{aoi}) and 
                            ContentDate/Start gt {self.start}T00:00:00.000Z and 
                            ContentDate/Start lt {self.end}T00:00:00.000Z""").json()
        
        # Retrieves scenes corresponding to the constraints  
        self.list_product = pd.DataFrame.from_dict(json['value'])
        assert len(self.list_product) != 0, 'No data found for your request'

        # Generates various outputs in dataframe format
        self.list_product = self.list_product[[self.product in p for p in self.list_product['S3Path']]]
        self.list_prod_id = list(self.list_product['Id'])
        self.list_prod_name = list(self.list_product['Name'])

        self.print_msg(f'Finds products matching the query ({len(self.list_prod_id)} products)')
        return self.list_prod_id

    def login(self, username: str, password: str):
        data = {
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
            }
        try:
            url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            r = requests.post(url, data=data)
            r.raise_for_status()
        except Exception:
            raise Exception(
                f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
                )
        self.tokens = r.json()["access_token"]
        self.print_msg('Log to API (https://identity.dataspace.copernicus.eu/)')

    def download_prod(self,
                      prod_id: str,
                      prod_name: str,
                      token_keys,
                      zip_format: bool = False):
        # Initialize session for download
        session = requests.Session()
        session.headers.update({'Authorization': f'Bearer {token_keys}'})
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"

        # Try to request server
        response = session.get(url, allow_redirects=False)
        niter = 0
        while response.status_code in (301, 302, 303, 307) and niter < 15:
            if response.status_code//100 == 5:
                raise ValueError(f'Got response code : {response.status_code}')
            if 'Location' not in response.headers:
                raise ValueError(f'status code : [{response.status_code}]')
            url = response.headers['Location']
            response = session.get(url, allow_redirects=False)
            niter += 1

        # Check if the file already exists
        file_mode = "wb"
        downloaded_bytes = 0
        os.makedirs(self.save_dir, exist_ok=True)
        filesize = int(response.headers.get("Content-Length"))

        if zip_format:
            local_filename = os.path.join(str(self.save_dir), prod_name+".zip")
        else:
            local_filename = os.path.join(str(self.save_dir), "tmp.zip")

        file_exists = os.path.exists(local_filename)
        if file_exists:
            return local_filename

        # Download file
        response = self.request_get(session, url, verify=False, allow_redirects=True)
        pbar = tqdm(total=filesize, unit_scale=True, unit="B", desc=f"Downloading {prod_name[:10]}",
                    unit_divisor=1024, initial=downloaded_bytes, leave=False)
        with open(local_filename, file_mode) as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)
        
        # Uncompress file if required
        if not zip_format:
            zip_file = ZipFile(local_filename, 'r')
            zip_file.extractall(self.save_dir)
            os.remove(local_filename)

    def get(self, list_id: list = None, zip_format: bool = False):
        # Prepare the progress bar
        self.print_msg('Start downloading')
        terminal_width, _ = shutil.get_terminal_size()
        if list_id:
            pbar = tqdm(list_id, ncols=terminal_width)
        else:
            pbar = tqdm(self.list_prod_id, ncols=terminal_width)
        pbar.set_description("Download Sentinel files")

        # Download each scene
        output_path = []
        for prod_id in pbar:
            name = str(self.list_product[self.list_product['Id'] == prod_id]['Name'].iloc[0])
            if len(name) == 0:
                print(f"[Warning] id={prod_id} not in available product list, product hasn't been downloaded")

            self.download_prod(prod_id, name, self.tokens, zip_format)
            if zip_format:
                output_path.append(os.path.join(self.save_dir, name+'.zip'))
            else:
                output_path.append(os.path.join(self.save_dir, name))

        self.print_msg(f'All {len(pbar)} files have been successfully downloaded')
        return output_path
