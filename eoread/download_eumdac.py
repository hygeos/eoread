from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import eumdac
from eoread import download
from eoread.fileutils import filegen
import shutil
from eoread.uncompress import uncompress



def query(collection, **kwargs):
    '''
    Query products with EUMDAC
    
    Collection: example 'EO:EUM:DAT:MSG:HRSEVIRI'

    kwargs: query arguments. Ex:
        title = 'MSG4-SEVI-MSG15-0100-NA-20221110081242.653000000Z-NA',
        dtstart = datetime.datetime(2022, 11, 10, 8, 0)
        dtend = datetime.datetime(2022, 11, 10, 8, 15)
    '''
    auth = download.get_auth('data.eumetsat.int')
    credentials = (auth['user'], auth['password'])   # key, secret
    token = eumdac.AccessToken(credentials)
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(collection)

    # Retrieve datasets that match our filter
    products = selected_collection.search(**kwargs)

    return products


def download_product(target, product):
    with TemporaryDirectory() as tmpdir, product.open() as fsrc:
        target_compressed = Path(tmpdir)/fsrc.name
        with open(target_compressed, mode='wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
            print(f'Download of product {product} finished.')
        uncompress(target_compressed, target.parent)


@filegen()
def download_eumdac(target: Path,
                    collections: Optional[list]=None):
    """
    Download a product on EUMETSAT data store
    
    collections: list of collections. Ex:
        - 'EO:EUM:DAT:MSG:HRSEVIRI' for SEVIRI
        - 'EO:EUM:DAT:0409' or 'EO:EUM:DAT:0577' for OLCI L1B FR
        - 'EO:EUM:DAT:0410' or 'EO:EUM:DAT:0578' for OLCI L1B RR
    """
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
