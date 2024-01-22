
from pathlib import Path
from datetime import datetime, timedelta

from src.download_sentinel import DownloadSentinel
from src.download_landsat import DownloadLandsat
from src.download_modis import DownloadModis
from src.download_eumdac import DownloadEumetsat


possible_collection  = ['SENTINEL-1',
                        'SENTINEL-2',
                        'SENTINEL-3',
                        'SENTINEL-5P',
                        'LANDSAT-5',
                        'LANDSAT-7',
                        'LANDSAT-8',
                        'LANDSAT-9',
                        "NASA-MODIS",
                        "EUMET-SEVIRI",
                        "EUMET-FCI",
                        "EUMET-RSS"]

class DownloadSatellite:

    def __init__(self, 
                 data_collection: str, 
                 save_dir: str | Path, 
                 start_date: str | datetime, 
                 lon_min: float = None, 
                 lon_max: float = None, 
                 lat_min: float = None, 
                 lat_max: float = None, 
                 bbox: list[float] = None,
                 point: tuple[float] = None,
                 inter_width: int | timedelta = None,
                 end_date: str | datetime = None,
                 product: str = None,
                 level: int = 1):
        
        self.data_collection = data_collection
        
        api = self._which_api()
        self.api = api( data_collection = data_collection, 
                        save_dir = save_dir, 
                        start_date = start_date, 
                        lon_min = lon_min, 
                        lon_max = lon_max, 
                        lat_min = lat_min, 
                        lat_max = lat_max, 
                        bbox = bbox,
                        point = point,
                        inter_width = inter_width,
                        end_date = end_date,
                        product = product,
                        level = level)

    def get_available(self):
        self.api.request_available()
        return self.api.list_product
    
    def download(self, list_id: list = None, compress_format: bool = False):
        return self.api.get(list_id, compress_format)
    
    def _which_api(self):        
        assert self.data_collection in possible_collection, \
        f'Invalid data collection [{self.data_collection}]'

        if "SENTINEL" in self.data_collection: 
            return DownloadSentinel
        if "LANDSAT" in self.data_collection: 
            return DownloadLandsat
        if "NASA" in self.data_collection: 
            return DownloadModis
        if "EUMET" in self.data_collection:
            return DownloadEumetsat
        