from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import eumdac
from tqdm import tqdm
from eoread import download_legacy as download
from eoread.utils.fileutils import filegen
import shutil
from eoread.utils.uncompress import uncompress as func_uncompress
import warnings


class DownloadEumetsat:
    def __init__(self, collection: str):
        """
        Download products from data.eumetsat.int

        Collections can be obtained with
            $ eumdac describe
        """
        auth = download.get_auth('data.eumetsat.int')
        credentials = (auth['user'], auth['password'])   # key, secret
        token = eumdac.AccessToken(credentials)
        self.datastore = eumdac.DataStore(token)
        self.collection = collection
        self.selected_collection = self.datastore.get_collection(collection)

    def query(self, **kwargs):
        """
        Query products from data.eumetsat.int

        kwargs: query arguments. Example:
            title = 'MSG4-SEVI-MSG15-0100-NA-20221110081242.653000000Z-NA',
            dtstart = datetime.datetime(2022, 11, 10, 8, 0),
            dtend = datetime.datetime(2022, 11, 10, 8, 15),
            geo = Point(lon, lat),
        
        This method can be decorated by cache_json for storing the outputs.
        Example:
            cache_json('cache_result.json')(dld.query)(title='MSG4...')
        """
        # Retrieve datasets that match our filter
        products = self.selected_collection.search(**kwargs)
        return [str(p) for p in products]

    def download(self, product_id: str, dir: Path, uncompress: bool=False) -> Path:
        """
        Download a product to directory

        product_id: 'S3A_OL_1_ERR____20231214T232432_20231215T000840_20231216T015921_2648_106_358______MAR_O_NT_002.SEN3'
        """
        product = self.datastore.get_product(
            product_id=product_id,
            collection_id=self.collection,
        )

        @filegen()
        def _download(target: Path):
            with TemporaryDirectory() as tmpdir:
                target_compressed = Path(tmpdir)/(product_id + '.zip')
                with product.open() as fsrc, open(target_compressed, mode='wb') as fdst:
                    pbar = tqdm(total=product.size*1e3, unit_scale=True, unit="B",
                                initial=0, unit_divisor=1024, leave=False)
                    pbar.set_description(f"Downloading {product_id}")
                    while True:
                        chunk = fsrc.read(1024)
                        if not chunk:
                            break
                        fdst.write(chunk)
                        pbar.update(len(chunk))
                print(f'Download of product {product} finished.')
                if uncompress:
                    func_uncompress(target_compressed, target.parent)
                else:
                    shutil.move(target_compressed, target.parent)

        target = dir/(product_id if uncompress else (product_id + '.zip'))

        _download(target)

        return target


def query(collection, **kwargs):
    '''
    Query products with EUMDAC
    
    Collection: example 'EO:EUM:DAT:MSG:HRSEVIRI'

    kwargs: query arguments. Ex:
        title = 'MSG4-SEVI-MSG15-0100-NA-20221110081242.653000000Z-NA',
        dtstart = datetime.datetime(2022, 11, 10, 8, 0)
        dtend = datetime.datetime(2022, 11, 10, 8, 15)
    '''
    warnings.warn(
        "This function is deprecated, please use class `DownloadEumetsat`",
        DeprecationWarning,
    )
    auth = download.get_auth('data.eumetsat.int')
    credentials = (auth['user'], auth['password'])   # key, secret
    token = eumdac.AccessToken(credentials)
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(collection)

    # Retrieve datasets that match our filter
    products = selected_collection.search(**kwargs)

    return products


def download_product(target, product):
    warnings.warn(
        "This function is deprecated, please use class `DownloadEumetsat`",
        DeprecationWarning,
    )
    with TemporaryDirectory() as tmpdir, product.open() as fsrc:
        target_compressed = Path(tmpdir)/fsrc.name
        with open(target_compressed, mode='wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
            print(f'Download of product {product} finished.')
        func_uncompress(target_compressed, target.parent)


@filegen(if_exists='skip')
def download_eumdac(target: Path,
                    collections: Optional[list]=None):
    """
    Download a product on EUMETSAT data store
    
    collections: list of collections. Ex:
        - 'EO:EUM:DAT:MSG:HRSEVIRI' for SEVIRI
        - 'EO:EUM:DAT:0409' or 'EO:EUM:DAT:0577' for OLCI L1B FR
        - 'EO:EUM:DAT:0410' or 'EO:EUM:DAT:0578' for OLCI L1B RR
    """
    warnings.warn(
        "This function is deprecated, please use class `DownloadEumetsat`",
        DeprecationWarning,
    )
    if collections is None:
        if '-SEVI-' in target.name:
            collections = ['EO:EUM:DAT:MSG:HRSEVIRI']
        elif '_OL_1_EFR____' in target.name:
            collections = ['EO:EUM:DAT:0409', 'EO:EUM:DAT:0577']
        elif '_OL_1_ERR____' in target.name:
            collections = ['EO:EUM:DAT:0410', 'EO:EUM:DAT:0578']
        elif '_OL_2_WFR____' in target.name:
            collections = [
                'EO:EUM:DAT:0407', # https://data.eumetsat.int/data/map/EO:EUM:DAT:0407
                'EO:EUM:DAT:0592', # https://data.eumetsat.int/data/map/EO:EUM:DAT:0592
                'EO:EUM:DAT:0556', # https://data.eumetsat.int/data/map/EO:EUM:DAT:0556
                ]
        else:
            raise ValueError()
    
    product = None
    for coll in collections:
        products = query(coll, title=target.name)

        if len(products) == 1:
            product = products.first()
        elif len(products) > 1:
            raise ValueError(f'Error, found {len(products)} products')
    
    if product is None:
        raise ValueError('Could not find any valid product.')

    download_product(target, product)
