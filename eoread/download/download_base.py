import time

from pathlib import Path
from datetime import datetime, timedelta
from netrc import netrc


class DownloadBase:

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

        # Check date format
        self.date_format = '%Y-%m-%d'
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, self.date_format)
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, self.date_format)

        # Compute temporal components
        if end_date:
            self.start = start_date
            self.end   = end_date
        elif inter_width:
            if isinstance(inter_width, int):
                inter_width = timedelta(days=inter_width)
            self.start = start_date - inter_width
            self.end   = start_date + inter_width
        else:
            raise ValueError('Should fill in end_date or inter_width')
        
        # Compute spatial components
        if None not in [lon_min, lon_max, lat_min, lat_max]:
            self.bbox = lon_min, lon_max, lat_min, lat_max
        elif bbox:
            self.bbox = bbox
        elif not point:
            raise ValueError('Should fill in lon/lat or bbox or point')
        self.point = point

        # Store other data
        self.data_collection = data_collection
        self.product = product
        self.save_dir = save_dir
        self.level = level

        assert level in [1,2], \
            f'Data level should be 1 or 2, get {level}'

    def get_collection(self):
        return NotImplemented

    def request_available(self):
        return NotImplemented

    def login(self, username: str, password: str):
        return NotImplemented

    def download_prod(self):
        return NotImplemented

    def get(self):        
        return NotImplemented
    
    def request_get(self, session, url, **kwargs):
        r = session.get(url, **kwargs)
        for i in range(10):
            try:
                raise_api_error(r)
            except RateLimitError:
                time.sleep(3)
                r = session.get(url, **kwargs)
        return r
    
    def _get_auth(self, name):
        """
        Returns a dictionary with credentials, using .netrc

        `name` is the identifier (= `machine` in .netrc). This allows for several accounts
        on a single machine.
        The url is returned as `account`
        """
        ret = netrc().authenticators(name)
        if ret is None:
            raise ValueError(
                f'Please provide entry "{name}" in ~/.netrc ; '
                f'example: machine {name} login <login> password <passwd> account <url>')
        (login, account, password) = ret

        return {'user': login,
                'password': password,
                'url': account}
    
    def print_msg(self, msg: str = ""):
        now = datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S')
        print(f'\033[92m[{now}]\033[0m INFO - {msg}')


def raise_api_error(response: dict):
    assert hasattr(response,'status_code')
    status = response.status_code

    if status == 401:
        raise UnauthorizedError
    if status == 404:
        raise FileNotFoundError
    if status == 429:
        raise RateLimitError
    
    if status//100 == 3:
        raise RedirectionError
    if status//100 == 4:
        raise InvalidParametersError
    if status//100 == 5:
        raise ServerError


class InvalidParametersError(Exception):
    """Provided parameters are invalid."""
    pass

class UnauthorizedError(Exception):
    """User does not have access to the requested endpoint."""
    pass

class RateLimitError(Exception):
    """Account does not support multiple requests at a time."""
    pass

class RedirectionError(Exception):
    """Account does not support multiple requests at a time."""
    pass

class ServerError(Exception):
    """The server failed to fulfil a request."""
    pass