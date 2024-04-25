import pytest 

from eoread.utils.save import to_netcdf, to_tif, to_img, to_gif
from eoread.reader.landsat9_oli import Level1_L9_OLI

from tempfile import TemporaryDirectory
from pathlib import Path


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

@pytest.mark.parametrize('ext',['png','jpg'])
def test_to_img_mask(level1_example, ext):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/('test.'+ext)
        to_img(level1_example['Rtoa'].isel(bands=0), 
               filename=outpath)

@pytest.mark.parametrize('ext',['png','jpg'])
def test_to_img_RGB(level1_example, ext):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/('test.'+ext)
        to_img(level1_example, 
               filename=outpath, 
               raster='Rtoa',
               rgb=[655,560,480])
        
def test_to_img_array(level1_example):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.gif'
        to_img(array=level1_example['Rtoa'].isel(bands=0).values,
               filename=outpath)
        
def test_to_gif_ds(level1_example):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.gif'
        to_gif(ds=level1_example,
               filename=outpath,
               raster='Rtoa',
               time_dim='bands')
