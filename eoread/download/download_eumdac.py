import eumdac
import requests
import shutil
import os

from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile 
from datetime import datetime, timedelta

from eoread.download.download_base import DownloadBase, UnauthorizedError


class DownloadEumetsat(DownloadBase):

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

        self.url_api       = 'https://data.eumetsat.int/'
        credentials = self._get_auth("data.eumetsat.int")
        self.client_id     = credentials['user']
        self.client_secret = credentials['password']
        self.login(self.client_id,self.client_secret)
        self.request_available()

    def request_available(self):
        datastore  = eumdac.DataStore(self.token)
        collection = self._get_dataset_name()
        selected_collection = datastore.get_collection(collection)
        self.product = list(selected_collection.search(
            # bbox = self.bbox,
            dtstart = self.start,
            dtend = self.end
        ))
        self.list_prod_name = [str(prod) for prod in self.product]
        assert len(self.product) != 0, 'No data found according to your request'
        self.print_msg(f'Finds products matching the query ({len(self.product)} products)')

    def login(self, username: str, password: str):
        credentials = (username, password)
        self.token = eumdac.AccessToken(credentials)
        try:
            if self.token.expiration < datetime.now():
                raise UnauthorizedError("Tokens has expired. Please refresh on https://api.eumetsat.int/api-key/#")
        except requests.exceptions.HTTPError:
            raise UnauthorizedError("Invalid Credentials")  
        self.print_msg(f'Log to API ({self.url_api})')

    def download_prod(self, product):
        pbar = tqdm(total=product.size*1e3, unit_scale=True, unit="B",
                    initial=0, unit_divisor=1024, leave=False)
        pbar.set_description(f"Downloading {str(product)[5:15]}")
        os.makedirs(self.save_dir, exist_ok=True)
        with product.open() as fsrc, open(os.path.join(self.save_dir,fsrc.name), mode='wb') as fdst:
            while True:
                chunk = fsrc.read(1024)
                if not chunk:
                    break
                fdst.write(chunk)
                pbar.update(len(chunk))
        
        return os.path.join(self.save_dir,fsrc.name)

    def get(self, list_id: list = None, zip_format: bool = False): 
        self.print_msg('Start downloading')
        terminal_width, _ = shutil.get_terminal_size()
        if list_id:
            pbar = tqdm(list_id, ncols=terminal_width)
        else:
            pbar = tqdm(self.product, ncols=terminal_width)
        pbar.set_description("Download Eumetsat files")

        output_path = []       
        for product in pbar:
            filename = self.download_prod(product)
            if zip_format:
                output_path.append(filename)
            else:
                zip_file = ZipFile(filename, 'r')
                zip_file.extractall(self.save_dir)
                os.remove(filename)
                output_path.append(os.path.join(self.save_dir, filename[:-4]))
        self.print_msg(f'All {len(pbar)} files have been successfully downloaded')
        return output_path
    
    def _get_dataset_name(self):
        if self.data_collection == 'EUMET-SEVIRI':
            collec_name = 'EO:EUM:DAT:MSG:HRSEVIRI'
        if self.data_collection == 'EUMET-RSS':
            collec_name = 'EO:EUM:DAT:MSG:MSG15-RSS'
        if self.data_collection == 'EUMET-OLCI-FR':
            collec_name = ['EO:EUM:DAT:0409', 'EO:EUM:DAT:0577']
        if self.data_collection == 'EUMET-OLCI-RR':
            collec_name = ['EO:EUM:DAT:0410', 'EO:EUM:DAT:0578']
        if self.data_collection == 'EUMET-FCI':
            collec_name = ''
        
        return collec_name