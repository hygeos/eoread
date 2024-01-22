#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Landsat-9 OLI reader

Example:
    l1 = Level1_L9_OLI('LC09_L1TP_014034_20220618_20230411_02_T1/')

Data access:
    * https://earthexplorer.usgs.gov/
    * https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1
'''

import os
from glob import glob
import datetime
import tempfile
import numpy as np
import xarray as xr
import rioxarray as rio
import rasterio
import dask.array as da
import pyproj
try:
    from osgeo import gdal, osr
    import osgeo
    gdal_major_version = int(osgeo.__version__.split('.')[0])
except ModuleNotFoundError:
    osr = None
    gdal = None
    gdal_major_version = None
from .. import common
from ..utils.naming import naming
from ..utils.tools import merge
from ..raster import ArrayLike_GDAL


PYPROJ_VERSION = int(pyproj.__version__.split('.')[0])


bands_oli  = np.array([440, 480, 560, 655, 865, 1375, 1610, 2200, 11000, 12000])
bands_vis = bands_oli[bands_oli < 3000]
bands_tir  = bands_oli[bands_oli > 3000]

band_index = { # Bands                                - wavelength (um) - resolution (m)
    440: 1,    # Band 1  - Coastal aerosol	            0.43 - 0.45	      30
    480: 2,    # Band 2  - Blue	                        0.45 - 0.51	      30
    560: 3,    # Band 3  - Green	                    0.53 - 0.59	      30
    655: 4,    # Band 4  - Red	                        0.64 - 0.67	      30
    865: 5,    # Band 5  - Near Infrared (NIR)	        0.85 - 0.88	      30
    1610: 6,   # Band 6  - SWIR 1	                    1.57 - 1.65	      30
    2200: 7,   # Band 7  - SWIR 2	                    2.11 - 2.29	      30
    # 580 : 8,   # Band 8  - Panchromatic	                0.50 - 0.68	      15
    1375: 9,   # Band 9  - Cirrus	                    1.36 - 1.38	      30
    11000:10,  # Band 10 - Thermal Infrared (TIRS) 1	10.60 - 11.19	  100 * (30)
    12000:11,  # Band 11 - Thermal Infrared (TIRS) 2	11.50 - 12.51	  100 * (30)
    }

# Central wavelength
# from Ball_BA_RSR.v1.2.xlsx
center_wavelengths = {
    440: 442.96,
    480: 482.04,
    560: 561.41,
    655: 654.59,
    865: 864.67,
    1610: 1608.86,
    2200: 2200.73,
    # 580: 581,
    1375: 1373.43,
    11000:10895,
    12000:12050,
}


def Level1_L9_OLI(dirname,
                  l9_angles=None,
                  radiometry='reflectance',
                  split=False,
                  chunks=500,
                  use_gdal=False,
                  angle_data=True,
                  ):
    '''
    Landsat-9 OLI reader.

    Arguments:
        dirname: name of the directory containing the Landsat9/OLI product
                 (Example: 'LC09_L1TP_014034_20220618_20230411_02_T1/')
        l9_angles: executable name of l9_angles program (ex: 'l9_angles/l9_angles'), used to generate the angles
                files automatically when missing, with the following command:
            l9_angles LC08_..._ANG.txt BOTH 1 -b 1
            l9_angles is available at:
            https://www.usgs.gov/land-resources/nli/landsat/solar-illumination-and-sensor-viewing-angle-coefficient-files

            It can be compiled with the following commands:
                wget https://landsat.usgs.gov/sites/default/files/documents/L9_ANGLES_2_7_0.tgz
                tar xzf L9_ANGLES_2_7_0.tgz
                rm -fv L9_ANGLES_2_7_0.tgz
                cd l9_angles
                make
                cd ..
        radiometry: str
            'radiance' or 'reflectance'
        split: boolean
            whether the wavelength dependent variables should be split in multiple 2D variables

    Returns a xr.Dataset
    '''
    ds = xr.Dataset()

    # Read metadata
    data_mtl = read_metadata(dirname)

    # get datetime
    d = data_mtl['IMAGE_ATTRIBUTES']['DATE_ACQUIRED']
    t = datetime.datetime.strptime(
        data_mtl['IMAGE_ATTRIBUTES']['SCENE_CENTER_TIME'][:8],
        '%H:%M:%S')
    ds.attrs[naming.datetime] = datetime.datetime.combine(
        d,
        datetime.time(t.hour, t.minute, t.second)
    ).isoformat()

    read_coordinates(ds, dirname, chunks, use_gdal)
    if angle_data:
        read_geometry(ds, dirname, l9_angles)
    ds = read_radiometry(
        ds, dirname, split, data_mtl, radiometry, chunks, use_gdal)

    # add center wavelengths
    ds[naming.wav] = xr.DataArray(
        da.from_array(np.array([center_wavelengths[b] for b in bands_vis],
                               dtype='float32')),
        dims=(naming.bands),
    )
    ds[naming.wav_tir] = xr.DataArray(
        da.from_array(np.array([center_wavelengths[b] for b in bands_tir],
                               dtype='float32')),
        dims=(naming.bands_tir),
    )

    # add flags
    ds[naming.flags] = xr.zeros_like(ds[naming.lat],
                                     dtype=naming.flags_dtype)

    # other attributes
    ds.attrs[naming.platform] = 'Landsat9'
    ds.attrs[naming.sensor] = 'OLI'
    ds.attrs[naming.product_name] = os.path.basename(os.path.abspath(dirname))
    ds.attrs[naming.input_directory] = os.path.dirname(os.path.abspath(dirname))


    return ds.unify_chunks()


def read_metadata(dirname):
    files_mtl = glob(os.path.join(dirname, 'LC*_MTL.txt'))
    if len(files_mtl) == 0:
        files_mtl = glob(os.path.join(dirname, 'LC*_MTL.xml'))
        assert len(files_mtl) == 1
        file_mtl = files_mtl[0]
        data_mtl = read_meta_xml(file_mtl)['LANDSAT_METADATA_FILE']
    else:
        assert len(files_mtl) == 1
        file_mtl = files_mtl[0]
        data_mtl = read_meta(file_mtl)['LANDSAT_METADATA_FILE']

    return data_mtl


def read_coordinates(ds, dirname, chunks, use_gdal):
    '''
    read lat/lon
    '''
    ds[naming.lat] = common.DataArray_from_array(
        LATLON(use_gdal=use_gdal)(dirname, 'lat'),
        naming.dim2,
        chunks=chunks,
    )
    ds[naming.lon] = common.DataArray_from_array(
        LATLON(use_gdal=use_gdal)(dirname, 'lon'),
        naming.dim2,
        chunks=chunks,
    )
    ds.attrs[naming.totalheight] = ds.rows.size
    ds.attrs[naming.totalwidth] = ds.columns.size


def gen_l9_angles(dirname, l9_angles=None):
    print(f'Geometry file is missing in {dirname}, generating it with {l9_angles}...')
    angles_txt_file = glob(os.path.join(dirname, 'LC09_*_ANG.txt'))
    assert len(angles_txt_file) == 1
    assert l9_angles is not None
    assert os.path.exists(l9_angles)
    path_exe = os.path.abspath(l9_angles)
    path_angles = os.path.abspath(angles_txt_file[0])
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = f'cd {tmpdir} ; {path_exe} {path_angles} BOTH 1 -b 1'
        os.system(cmd)
        angle_files = os.path.join(tmpdir, '*')
        os.system(f'cp -v {angle_files} {dirname}')


def read_geometry(ds, dirname, l9_angles):
    filenames_saa = glob(os.path.join(dirname, 'LC*_SAA.TIF'))
    filenames_sza = glob(os.path.join(dirname, 'LC*_SZA.TIF'))

    # if ((not filenames_saa) or (not filenames_sza)) and (l9_angles is not None):
    #     gen_l9_angles(dirname, l9_angles)
    #     filenames_sensor = glob(os.path.join(dirname, 'LC*_sensor_B01.img'))

    # read sensor angles
    assert len(filenames_saa) == 1, \
        f'Error, sensor angles file missing in {dirname} ({str(filenames_saa)})'
    assert len(filenames_sza) == 1, \
        f'Error, sensor angles file missing in {dirname} ({str(filenames_sza)})'

    data_saa = da.from_array(
        rasterio.open(filenames_saa[0]).read(),
        meta=np.array([], 'int16')
        )
    
    data_sza = da.from_array(
        rasterio.open(filenames_sza[0]).read(),
        meta=np.array([], 'int16')
        )

    ds[naming.sza] = (naming.dim2, (data_sza[0, :, :]/100.).astype('float32'))
    ds[naming.saa] = (naming.dim2, (data_saa[0, :, :]/100.).astype('float32'))


    # read solar angles
    filenames_vaa = glob(os.path.join(dirname, 'LC*_VAA.TIF'))
    filenames_vza = glob(os.path.join(dirname, 'LC*_VZA.TIF'))

    assert len(filenames_vaa) == 1, \
        f'Error, sensor angles file missing in {dirname} ({str(filenames_vaa)})'
    assert len(filenames_vza) == 1, \
        f'Error, sensor angles file missing in {dirname} ({str(filenames_vza)})'

    data_vaa = da.from_array(
        rasterio.open(filenames_vaa[0]).read(),
        meta=np.array([], 'int16')
        )
    
    data_vza = da.from_array(
        rasterio.open(filenames_vza[0]).read(),
        meta=np.array([], 'int16')
        )

    ds[naming.vza] = (naming.dim2, (data_vza[0, :, :]/100.).astype('float32'))
    ds[naming.vaa] = (naming.dim2, (data_vaa[0, :, :]/100.).astype('float32'))


def read_radiometry(ds, dirname, split, data_mtl, radiometry, chunks, use_gdal):
    param = {'radiance':   (naming.Ltoa, naming.Ltoa_tir),
             'reflectance':(naming.Rtoa, naming.BT)}[radiometry]

    bnames = []
    for b in bands_vis:
        bname = (param[0]+'_{}').format(b)
        bnames.append(bname)
        ds[bname] = common.DataArray_from_array(
            TOA_READ(b, dirname, radiometry, data_mtl, use_gdal=use_gdal),
            naming.dim2,
            chunks=chunks,
        )
        ds[bname] /= da.cos(da.radians(ds.sza))

    if not split:
        ds = merge(ds, dim=naming.bands)
    
    bnames = []
    for b in bands_tir:
        bname = (param[1]+'_{}').format(b)
        bnames.append(bname)
        ds[bname] = common.DataArray_from_array(
            BT_READ(b, dirname, radiometry, data_mtl, use_gdal=use_gdal),
            naming.dim2,
            chunks=chunks,
        )

    if not split:
        ds = merge(ds, dim=naming.bands_tir)

    return ds


def LATLON(use_gdal=False):
    if use_gdal:
        return LATLON_GDAL
    else:
        return LATLON_NOGDAL


class LATLON_GDAL:
    def __init__(self, dirname, kind, dtype='float32'):
        '''
        kind: 'lat' or 'lon'
        '''
        self.kind = kind

        if osr is None:
            raise Exception('Error, gdal is not available')

        files_B1 = glob(os.path.join(dirname, 'LC*_B1.TIF'))
        if len(files_B1) != 1:
            raise Exception('Invalid directory content ({})'.format(files_B1))
        file_B1 = files_B1[0]

        b1 = gdal.Open(file_B1)
        old_cs = osr.SpatialReference()
        if gdal_major_version >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            old_cs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        old_cs.ImportFromWkt(b1.GetProjectionRef())

        # create the new coordinate system
        wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""
        new_cs = osr.SpatialReference()
        if gdal_major_version >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            new_cs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        new_cs.ImportFromWkt(wgs84_wkt)

        # create a transform object to convert between coordinate systems
        self.transform = osr.CoordinateTransformation(old_cs, new_cs)

        # get the point to transform, pixel (0,0) in this case
        width = b1.RasterXSize
        height = b1.RasterYSize
        gt = b1.GetGeoTransform()

        X0, X1 = (0, width-1)
        Y0, Y1 = (0, height-1)

        # generally:
        # Xgeo0 = gt[0] + X0*gt[1] + Y0*gt[2]
        # Ygeo0 = gt[3] + X0*gt[4] + Y0*gt[5]
        # Xgeo1 = gt[0] + X1*gt[1] + Y0*gt[2]
        # Ygeo1 = gt[3] + X1*gt[4] + Y0*gt[5]
        # Ygeo2 = gt[3] + X0*gt[4] + Y1*gt[5]
        # Xgeo2 = gt[0] + X0*gt[1] + Y1*gt[2]
        # Xgeo3 = gt[0] + X1*gt[1] + Y1*gt[2]
        # Ygeo3 = gt[3] + X1*gt[4] + Y1*gt[5]

        # this simplifies because gt[2] == gt[4] == 0
        assert gt[2] == 0
        assert gt[4] == 0
        Xmin = gt[0] + X0*gt[1]
        Xmax = gt[0] + X1*gt[1]
        Ymin = gt[3] + Y0*gt[5]
        Ymax = gt[3] + Y1*gt[5]

        self.shape = (height, width)
        self.ndim = 2
        self.X = np.linspace(Xmin, Xmax, width)
        self.Y = np.linspace(Ymin, Ymax, height)
        self.dtype = np.dtype(dtype)


    def __getitem__(self, keys):
        x = self.X[keys[1]]
        y = self.Y[keys[0]]

        sx = (len(x),) if hasattr(x, '__len__') else ()
        sy = (len(y),) if hasattr(y, '__len__') else ()

        XY = np.array(np.meshgrid(x, y))
        XY = np.moveaxis(XY, 0, -1)

        # get the coordinates in lat long
        latlon = np.array(
            self.transform.TransformPoints(XY.reshape((-1, 2))),
            dtype=self.dtype)
        assert latlon.dtype == self.dtype
        if self.kind == 'lat':
            return latlon[:, 1].reshape(sy+sx)
        else:
            assert self.kind == 'lon'
            return latlon[:, 0].reshape(sy+sx)


class LATLON_NOGDAL:
    def __init__(self, dirname, kind, dtype='float32'):
        self.kind = kind

        files_B1 = glob(os.path.join(dirname, 'LC*_B1.TIF'))
        if len(files_B1) != 1:
            raise Exception('Invalid directory content ({})'.format(files_B1))
        file_B1 = files_B1[0]

        data = rasterio.open(file_B1)

        height = data.height
        width = data.width
        self.shape = (height, width)

        gt = data.transform
        X0, X1 = (0, width-1)
        Y0, Y1 = (0, height-1)
        assert gt[1] == 0
        assert gt[3] == 0
        Xmin = gt[2] + X0*gt[0]
        Xmax = gt[2] + X1*gt[0]
        Ymin = gt[5] + Y0*gt[4]
        Ymax = gt[5] + Y1*gt[4]

        self.X = np.linspace(Xmin, Xmax, width)
        self.Y = np.linspace(Ymin, Ymax, height)
        self.dtype = np.dtype(dtype)
        
        self.latlon = pyproj.Proj("EPSG:4326")   # WGS84
        self.utm = pyproj.Proj(data.crs)

        if PYPROJ_VERSION >= 2:
            # see https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
            self.transformer = pyproj.Transformer.from_proj(self.utm, self.latlon)
        else:
            self.transformer = None

    def __getitem__(self, keys):
        x = self.X[keys[1]]
        y = self.Y[keys[0]]
        sx = (len(x),) if hasattr(x, '__len__') else ()
        sy = (len(y),) if hasattr(y, '__len__') else ()


        X, Y = np.meshgrid(x, y)

        if PYPROJ_VERSION >= 2:
            lat, lon = self.transformer.transform(X, Y)
        else:
            lat, lon = pyproj.transform(self.utm, self.latlon, X, Y)


        if self.kind == 'lat':
            return lat.astype(self.dtype).reshape(sy+sx)
        else:
            return lon.astype(self.dtype).reshape(sy+sx)


class TOA_READ:
    '''
    An array-like to read Landsat-8 OLI TOA reflectance
    (have to be divided by cos(sza) - radiance and reflectance
    in L1 are independent of the geometry)

    Arguments:
        b: band identifier (440, 480, 560, 655, 865)
        kind: 'radiance' or 'reflectance'
    '''
    def __init__(self,
                 b,
                 dirname,
                 radiometry='reflectance',
                 data_mtl=None,
                 use_gdal=False,
                 dtype='float32'):
        if data_mtl is None:
            data_mtl = read_metadata(dirname)

        if radiometry == 'radiance':
            param_mult = 'RADIANCE_MULT_BAND_{}'
            param_add = 'RADIANCE_ADD_BAND_{}'
        elif radiometry == 'reflectance':
            param_mult = 'REFLECTANCE_MULT_BAND_{}'
            param_add = 'REFLECTANCE_ADD_BAND_{}'
        else:
            raise Exception('TOA_READ: `kind` should be `radiance` or `reflectance`')

        self.filename = os.path.join(
            dirname,
            data_mtl['PRODUCT_CONTENTS']['FILE_NAME_BAND_{}'.format(band_index[b])])

        if use_gdal:
            self.data = ArrayLike_GDAL(self.filename)
        else:
            self.data = rio.open_rasterio(self.filename).isel(band=0)

        self.M = data_mtl['LEVEL1_RADIOMETRIC_RESCALING'][param_mult.format(band_index[b])]
        self.A = data_mtl['LEVEL1_RADIOMETRIC_RESCALING'][param_add.format(band_index[b])]
        self.data  = self.M * self.data + self.A
        
        self.dtype = np.dtype(dtype)
        self.shape = self.data.shape
        self.ndim = 2

    def __getitem__(self, keys):
        data = self.data[keys]
        return data.astype(self.dtype)

class BT_READ:
    '''
    An array-like to read Landsat-8 OLI TOA reflectance
    (have to be divided by cos(sza) - radiance and reflectance
    in L1 are independent of the geometry)

    Arguments:
        b: band identifier (440, 480, 560, 655, 865)
        kind: 'radiance' or 'reflectance'
    '''
    def __init__(self,
                 b,
                 dirname,
                 radiometry='reflectance',
                 data_mtl=None,
                 use_gdal=False,
                 dtype='float32'):
        if data_mtl is None:
            data_mtl = read_metadata(dirname)

        param_mult = 'RADIANCE_MULT_BAND_{}'
        param_add = 'RADIANCE_ADD_BAND_{}'

        self.filename = os.path.join(
            dirname,
            data_mtl['PRODUCT_CONTENTS']['FILE_NAME_BAND_{}'.format(band_index[b])])

        if use_gdal:
            self.data = ArrayLike_GDAL(self.filename)
        else:
            self.data = rio.open_rasterio(self.filename).isel(band=0)
        
        self.M = data_mtl['LEVEL1_RADIOMETRIC_RESCALING'][param_mult.format(band_index[b])]
        self.A = data_mtl['LEVEL1_RADIOMETRIC_RESCALING'][param_add.format(band_index[b])]
        self.data  = self.M * self.data + self.A
        
        if radiometry == 'reflectance':
            self.K1 = data_mtl['LEVEL1_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_{}'.format(band_index[b])]
            self.K2 = data_mtl['LEVEL1_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_{}'.format(band_index[b])]
            self.data = self.K2/np.log(self.K1/self.data + 1)

        self.dtype = np.dtype(dtype)
        self.shape = self.data.shape
        self.ndim = 2

    def __getitem__(self, keys):
        data = self.data[keys]
        return data.astype(self.dtype)

def node(raw, data):
    if 'END_GROUP' in raw[0]:
        return raw[1:] 

    if 'GROUP' in raw[0]:
        key = raw[0].split('=')[1].strip()
        data[key] = {}
        raw = node(raw[1:], data[key])
        return raw

    else:
        key, value, raw = leaf(raw)
        data[key] = value
        raw = node(raw[1:], data)

    return raw

def leaf(raw):
    key = raw[0].split('=')[0].strip()
    value = raw[0].split('=')[1].strip()

    if value[0] == '"': # string
        value = value[1:-1]
    elif value[0] == '(':
        tmp = [float(a) for a in value[1:-1].split(',')]

        while value[-1] != ')': # list
            raw = raw[1:]
            value = raw[0].strip()
            tmp += [float(a) for a in value[1:-1].split(',')]

        value = tmp
    else:
        try:
            if ':' in value:
                pass
            elif '.' in value: #float
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            value = np.datetime64(value).astype(datetime.datetime)

    return key, value, raw

def parser(raw):
    data = {}
    subdata = data

    if 'GROUP' in raw[0] and 'GROUP' in raw[1]:
        key = raw[0].split('=')[1].strip()
        data[key] = {}
        subdata = data[key]
        raw = raw[1:]

    while len(raw)!=0:
        raw = node(raw, subdata)
        if raw[0][:3]=='END':
            break

    return data

def read_meta(filename):
    '''
    A parser for Landsat8 metadata and angles file in ODL (Object Desription Language)
    '''
    with open(filename) as pf:
        raw = pf.readlines()

    data = parser(raw)

    return data

def read_meta_xml(filename):
    data = 0

    return data