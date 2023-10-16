from pydap.cas.urs import setup_session
from pydap.client import open_url

from datetime import date
import subprocess

from eoread import download as dl

class Merra2Parser:
    """
    Web-scrap and parses NASA's MERRA2 OPeNDAP website to map data:
    products, their version, and the variables each one contains 
    """
    # TODO remove lat, lon and time from variables ?
    auth = dl.get_auth('urs.earthdata.nasa.gov')
    base_opendap_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/'
    
    def get_products_vers(self) -> dict:
        """
        Returns a dictionnary of product associated with their current versions
        """
        cmd = f"curl -s {Merra2Parser.base_opendap_url} \
            | grep 'contents.html\">M2' | cut -d '>' -f 2 | cut -d '/' -f 1"
            
        status, output = subprocess.getstatusoutput(cmd) # execute shell cmd
        
        res =  [item.split('.', 1) for item in output.split('\n')]
        
        # transform list of list to a dictionnary
        return {item[0]: item[1] for item in res}
        
    
    def _get_variable_names(self, product: str, version: str, dat: date, file: str) -> list[str]:
        """
        Returns a list of the variable contained in the dataset
        uses OPeNDAP instead of xarray because it is faster for this use
        """
        
        dataset_url = Merra2Parser.base_opendap_url + product + '.' + version + '/' \
                    + dat.strftime('%Y/%m/') + file # url to the month folder of the product
                    
        username = self.auth['user']
        password = self.auth['password']
        dataset = {}
        try:
            session = setup_session(username, password, check_url=dataset_url)
            dataset = open_url(dataset_url, session=session)
        except AttributeError as e:
            print('Error:', e)
            print('Please verify that the dataset URL points to an OPeNDAP server, the OPeNDAP server is accessible, or that your username and password are correct.')
            
        res = list(dataset)
        
        # remove coordinates
        if 'lat'  in res: res.remove('lat')
        if 'lon'  in res: res.remove('lon')
        if 'time' in res: res.remove('time')
        
        return res


    def _get_product_generic_filename(self, product: str, version: str, dat: date):
        """
        Parse an OPeNDAP folder from MERRA-2 and return the generic filename,
        ready to be formated with a date
        
        dat: only year and month will be used
        """
        
        url = Merra2Parser.base_opendap_url + product + '.' + version + '/' + dat.strftime('%Y/%m/') # url to the month folder of the product

        cmd = f"curl -s {url} | grep -E '.*\"name\": \"MERRA2'"                 # only get html lines where we can extract the filename
        status, output = subprocess.getstatusoutput(cmd)                        # execute request

        file_name = output.strip().split('\n')[0].split(' ')[1].split('\"')[1]  # parsing
        
        parts = file_name.split('.')                                            # extract the date
        generic_name = parts[0] + '.' + parts[1] + '.%s.' + parts[3]            # replace it by generic '%s'

        return generic_name
        
    
    def get_products_specs(self, d: date):
        """
        Returns a dictionnary containing every MERRA-2 product,
        its version, name, and variables contained
        """
        products_vers = self.get_products_vers() # dictionnary of product: version
        
        cpt = 0                     # used for better IO message
        nbr = len(products_vers)    # used for better IO message
        specs = {}
        
        print(f'Parsing MERRA-2 products: [0/{nbr}]')
        for product in products_vers:   # 
            cpt += 1
            
            # get every attribute
            name = product
            version = products_vers[product]
            filename = self._get_product_generic_filename(product, version, d)
            variables = self._get_variable_names(product, version, d, filename % d.strftime('%Y%m%d'))
            
            # add values to the dictionnary
            specs[product] = {'name': name, 'version': version, 'generic_filename': filename, 'variables': variables}
            print(f'Parsing MERRA-2 products: [{cpt}/{nbr}] \t {name}')
        return specs
    
    
