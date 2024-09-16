import pytest 

from core.save import to_netcdf
from eoread.utils.save_aux import to_tif, to_img, to_gif
from eoread.reader.landsat9_oli import Level1_L9_OLI

from tempfile import TemporaryDirectory
from imageio.v2 import imread 
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


@pytest.fixture()
def level1_example():
    filepath = '/archive2/proj/QTIS_TRISHNA/L8L9/USA/LC09_L1TP_014034_20220618_20230411_02_T1'
    return Level1_L9_OLI(filepath)


@pytest.mark.skip('to_netcdf does not support xr.DataArray')
def test_to_netcdf_dataarray(level1_example):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.nc'
        to_netcdf(level1_example['Rtoa'], filename=outpath)
        
def test_to_netcdf_dataset(level1_example):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.nc'
        to_netcdf(level1_example, filename=outpath)

@pytest.mark.parametrize('compress',[True])
def test_to_tiff_dataset(level1_example, compress):
    ds = level1_example[['Rtoa','latitude','longitude']]
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.tif'
        to_tif(ds,
               filename=outpath,
               compressor=compress)
        
@pytest.mark.parametrize('compress',[True])
def test_to_tiff_dataarray(level1_example, compress):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.tif'
        to_tif(level1_example, 
               filename=outpath, 
               raster='Rtoa',
               compressor=compress)
    xr.open_dataarray(outpath).plot.imshow()

@pytest.mark.parametrize('ext',['png','jpg'])
def test_to_img_mask(level1_example, ext):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/('test.'+ext)
        to_img(level1_example['Rtoa'].isel(bands=0), 
               filename=outpath)
    img = imread(outpath)
    plt.imshow(img)

@pytest.mark.parametrize('ext',['png','jpg'])
def test_to_img_RGB(level1_example, ext):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/('test.'+ext)
        to_img(level1_example, 
               filename=outpath, 
               raster='Rtoa',
               rgb=[655,560,480])
    img = imread(outpath)
    plt.imshow(img)
        
def test_to_img_array(level1_example):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.gif'
        to_img(array=level1_example['Rtoa'].isel(bands=0).values,
               filename=outpath)
    img = imread(outpath)
    plt.imshow(img)
        
def test_to_gif_ds(level1_example):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.gif'
        to_gif(ds=level1_example,
               filename=outpath,
               raster='Rtoa',
               time_dim='bands')
