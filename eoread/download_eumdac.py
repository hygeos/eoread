from pathlib import Path
from tempfile import TemporaryDirectory
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
def download_eumdac(target: Path, collection=None):
    if collection is None:
        if '-SEVI-' in  target.name:
            collection = 'EO:EUM:DAT:MSG:HRSEVIRI'
        else:
            raise ValueError()
    
    products = query(collection, title=target.name)

    assert len(products) == 1
    product = products.first()

    download_product(target, product)
