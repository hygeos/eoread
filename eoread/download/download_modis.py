import requests

from pathlib import Path
from datetime import datetime, timedelta
# from modis_tools.resources import CollectionApi, GranuleApi
# from modis_tools.granule_handler import GranuleHandler

from eoread.download.download_base import DownloadBase


# BASED ON : https://github.com/fraymio/modis-tools/

class DownloadModis(DownloadBase):

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

        self.api_url       = "https://cmr.earthdata.nasa.gov"
        credentials = self._get_auth("earthdata.nasa.gov")
        self.client_id     = credentials['user']
        self.client_secret = credentials['password']
        self.login(self.client_id,self.client_secret)
        # self.get_collection()
        # self.request_available()
    
    def get_collection(self):
        r = self.request_get("/".join([self.api_url, "search", "collections"]))
        return r

        # collection_client = CollectionApi(session=self.session)
        # self.collections = collection_client.query(short_name="MOD13A1", version="061")
        # return [col.title for col in self.collections]

    def request_available(self):
        # Query the selected collection for granules
        granule_client = GranuleApi.from_collection(self.collections[0], session=self.session)
        self.list_product = granule_client.query(start_date=self.date_range[0], end_date=self.date_range[1], bounding_box=self.bbox)
        return self.list_product

    def login(self, username: str, password: str):
        self.session = requests.sessions.Session()
        # self.session.auth = requests.auth.HTTPBasicAuth(username, password)
        self.session.headers["Accept"] = "application/json"

    def get(self, list_id: list = None, tar_format: bool = False):
        if list_id is None:
            list_id = self.list_prod_id

        GranuleHandler.download_from_granules(list_id, self.session, path=self.save_dir)
