from core.monitor import Chrono
from datetime import timedelta
from core import log
import importlib
import argparse
import warnings

warnings.filterwarnings("ignore", message=".*Increasing number of chunks.*")


readers = {
    'ecostress' : 'Level1_ECOSTRESS',
    # 'goesng'    : 'Level1_GOESNG',
    'hypso'     : 'Level1_HYPSO',
    'landsat_oli' : 'Level1_OLI',
    'meris'     : 'Level1_MERIS',
    'modis'     : 'Level1_MODIS',
    'msi'       : 'Level1_MSI',
    # 'nasa'      : 'Level1_NASA',
    'olci'      : 'Level1_OLCI',
    'sgli'      : 'Level1_SGLI',
    'venus'     : 'Level1_VENUS',
}

parser = argparse.ArgumentParser('Check reader')
subs   = parser.add_subparsers(dest="cmd", required=True)
cmd_ls = subs.add_parser(name='list', help='list available readers')
time   = subs.add_parser(name='time', help='test reading time for all readers')
check  = subs.add_parser(name='test', help='test reader')
check.add_argument('reader', action="store", help="Reader to test")

args = parser.parse_args()

if args.cmd == 'list':
    
    log.info('Available readers are the followings :')
    log_format = ' * {}'
    for reader in readers.keys():
        log.info(log_format.format(reader))
        
if args.cmd == 'test':
    
    reader = args.reader
    module = importlib.import_module('eoread.legacy.'+reader)
    reader = getattr(module, readers[reader])
    sample = getattr(module, 'get_sample')
    
    filename = sample(1)
    with Chrono('reading time', unit='s'):
        l1 = reader(filename)

if args.cmd == 'time':
    
    c = Chrono('reading time', unit='s')
    for name, reader in readers.items():
        module = importlib.import_module('eoread.legacy.'+name)
        reader = getattr(module, reader)
        sample = getattr(module, 'get_sample')
        
        log.set_lvl(log.lvl.ERROR)
        filename = sample(1)
        c.reset()
        l1 = reader(filename)
        time = c.stop()
        
        log.set_lvl(log.lvl.INFO)
        if time < timedelta(seconds=1): color = log.rgb.green
        else: color = log.rgb.red
        log.info(color, name, ' --> ', str(time))