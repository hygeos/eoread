from datetime import datetime, date, time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from eoread.download.download_base import request_get
from eoread.download_legacy import get_auth
from eoread.utils.fileutils import filegen


collections = [
    'SENTINEL-1',
    'SENTINEL-2',
    'SENTINEL-3',
    'SENTINEL-5P',
]

class DownloadCDS:
    def __init__(self, collection: str):
        """
        Python interface to the Copernicus Data Space (https://dataspace.copernicus.eu/)

        Args:
            collection (str): collection name ('SENTINEL-2', 'SENTINEL-3', etc.)

        Example:
            cds = DownloadCDS('SENTINEL-2')
            # retrieve the list of products
            # using a json cache file to avoid reconnection
            ls = cache_json('query-S2.json')(cds.query)(
                dtstart=datetime(2024, 1, 1),
                dtend=datetime(2024, 2, 1),
                geo=Point(119.514442, -8.411750),
                name_contains='_MSIL1C_',
            )
            for p in ls:
                cds.download(p, <dirname>, uncompress=True)
        """
        assert collection in collections
        self.collection = collection
        self._login()

    def _login(self):
        """Login to copernicus dataspace with credentials storted in .netrc
        """
        auth = get_auth("dataspace.copernicus.eu")

        data = {
            "client_id": "cdse-public",
            "username": auth['user'],
            "password": auth['password'],
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
        print('Log to API (https://identity.dataspace.copernicus.eu/)')

    def query(
        self,
        dtstart: date|datetime,
        dtend: date|datetime,
        geo=None,
        cloudcover_thres: Optional[int]=None,
        name_contains: Optional[str] = None,
        other_attrs: Optional[list] = None,
    ):
        """
        Product query on the Copernicus Data Space

        Args:
            dtstart and dtend (datetime): start and stop datetimes
            geo: shapely geometry. Examples:
                Point(lon, lat)
                Polygon(...)
            cloudcover_thres: Optional[int]=None,
            name_contains (str): start of the product name
            other_attrs (list): list of other attributes to include in the output
                (ex: ['ContentDate', 'Footprint'])

        Note:
            This method can be decorated by cache_json for storing the outputs.
            Example:
                cache_json('cache_result.json')(cds.query)(...)
        """
        # https://documentation.dataspace.copernicus.eu/APIs/OData.html#query-by-name
        if isinstance(dtstart, date):
            dtstart = datetime.combine(dtstart, time(0))
        if isinstance(dtend, date):
            dtend = datetime.combine(dtend, time(0))
        query_lines = [
            f"""https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name 
                eq '{self.collection}' """,
            f'ContentDate/Start gt {dtstart.isoformat()}Z',
            f'ContentDate/Start lt {dtend.isoformat()}Z',
        ]

        if geo:
            query_lines.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{geo}')")

        if name_contains:
            query_lines.append(f"contains(Name, '{name_contains}')")

        if cloudcover_thres:
            query_lines.append(
                "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
                f"and att/OData.CSC.DoubleAttribute/Value le {cloudcover_thres})")

        json = requests.get(' and '.join(query_lines)).json()

        return [{"id": d["Id"], "name": d["Name"],
                 **{k: d[k] for k in (other_attrs or [])}}
                for d in json["value"]]

    def download(self, product: dict, dir: Path|str, uncompress: bool=True) -> Path:
        """Download a product from copernicus data space

        Args:
            product (dict): product definition
            dir (Path | str): _description_
            uncompress (bool, optional): _description_. Defaults to True.
        """
        if uncompress:
            target = Path(dir)/(product['name'])
            uncompress_ext = '.zip'
        else:
            target = Path(dir)/(product['name']+'.zip')
            uncompress_ext = None

        filegen(0, uncompress=uncompress_ext)(self._download)(target, product)

        return target

    def _download(
        self,
        target: Path,
        product: dict,
    ):
        """
        Wrapped by filegen
        """
        pbar = tqdm(total=0, unit_scale=True, unit="B",
                    unit_divisor=1024, leave=False)

        # Initialize session for download
        session = requests.Session()
        session.headers.update({'Authorization': f'Bearer {self.tokens}'})
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product['id']})/$value"

        # Try to request server
        pbar.set_description('Try to request server')
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

        # Download file
        filesize = int(response.headers["Content-Length"])
        response = request_get(session, url, verify=False, allow_redirects=True)
        pbar = tqdm(total=filesize, unit_scale=True, unit="B",
                    unit_divisor=1024, leave=False)
        pbar.set_description(f"Downloading {product['name']}")
        with open(target, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)
