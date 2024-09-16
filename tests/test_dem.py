
from eoread.reader.dem import SRTM, GTOPO30
from core.tools import xrcrop
from eoread.reader.landsat9_oli import Level1_L9_OLI


l1_path = '/archive2/proj/QTIS_TRISHNA/L8L9/USA/LC09_L1TP_014034_20220618_20230411_02_T1/'

def test_srtm():
    l1 = Level1_L9_OLI(l1_path)
    srtm = SRTM(missing=0)
    sub = xrcrop(srtm, latitude=l1.latitude, longitude=l1.longitude)
    sub.compute(scheduler='sync')

def test_gtopo():
    l1 = Level1_L9_OLI(l1_path)
    gtopo = GTOPO30(missing=0)
    sub = xrcrop(gtopo, latitude=l1.latitude, longitude=l1.longitude)
    sub.compute()