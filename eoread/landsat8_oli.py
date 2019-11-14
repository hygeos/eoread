#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Landsat-8 OLI reader

Example:
    l1 = Level1_L8_OLI('LC80140282017275LGN00/')

Data access:
    * https://earthexplorer.usgs.gov/
    * https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_TOA
'''

import os
from glob import glob
import datetime
import tempfile
import numpy as np
import xarray as xr
import dask.array as da
from osgeo import gdal
import osr
from . import common
from .naming import Naming
from . import eo



bands_oli = [440, 480, 560, 655, 865, 1375, 1610, 2200]
band_index = { # Bands - wavelength (um) - resolution (m)
    440: 1,    # Band 1 - Coastal aerosol	0.43 - 0.45	30
    480: 2,    # Band 2 - Blue	0.45 - 0.51	30
    560: 3,    # Band 3 - Green	0.53 - 0.59	30
    655: 4,    # Band 4 - Red	0.64 - 0.67	30
    865: 5,    # Band 5 - Near Infrared (NIR)	0.85 - 0.88	30
    1610: 6,   # Band 6 - SWIR 1	1.57 - 1.65	30
    2200: 7,   # Band 7 - SWIR 2	2.11 - 2.29	30
               # Band 8 - Panchromatic	0.50 - 0.68	15
    1375: 9,   # Band 9 - Cirrus	1.36 - 1.38	30
               # Band 10 - Thermal Infrared (TIRS) 1	10.60 - 11.19	100 * (30)
               # Band 11 - Thermal Infrared (TIRS) 2	11.50 - 12.51	100 * (30)
    }

def Level1_L8_OLI(dirname, l8_angles=None, radiometry='reflectance',
                  split=False, naming=Naming(), chunksize=(300, 400)):
    '''
    Landsat-8 OLI reader.

    Arguments:
        dirname: name of the directory containing the Landsat8/OLI product
                 (Example: 'LC80140282017275LGN00/')
        l8_angles: executable name of l8_angles program (ex: 'l8_angles/l8_angles'), used to generate the angles
                files automatically when missing, with the following command:
            l8_angles LC08_..._ANG.txt BOTH 1 -b 1
            l8_angles is available at:
            https://www.usgs.gov/land-resources/nli/landsat/solar-illumination-and-sensor-viewing-angle-coefficient-files

            It can be compiled with the following commands:
                wget https://landsat.usgs.gov/sites/default/files/documents/L8_ANGLES_2_7_0.tgz
                tar xzf L8_ANGLES_2_7_0.tgz
                rm -fv L8_ANGLES_2_7_0.tgz
                cd l8_angles
                make
                cd ..
        radiometry: 'radiance' or 'reflectance'
        split: (boolean) whether the wavelength dependent variables should be split in multiple 2D variables
        naming: a Naming instance, used for parameters naming consistency.
                Ex: custom naming: naming=Naming(Rtoa='RTOA')
        chunksize: dask arrays chunk sizes.

    Returns a xr.Dataset
    '''
    ds = xr.Dataset()

    # Read metadata
    data_mtl = read_metadata(dirname)

    # get datetime
    d = data_mtl['PRODUCT_METADATA']['DATE_ACQUIRED']
    t = datetime.datetime.strptime(
        data_mtl['PRODUCT_METADATA']['SCENE_CENTER_TIME'][:8],
        '%H:%M:%S')
    ds.attrs[naming.datetime] = datetime.datetime.combine(d, datetime.time(t.hour, t.minute, t.second))

    read_coordinates(ds, dirname, naming, chunksize)
    read_geometry(ds, dirname, l8_angles, naming, chunksize)
    ds = read_radiometry(
        ds, dirname, split, data_mtl, radiometry, naming, chunksize)

    return ds


def read_metadata(dirname):
    files_mtl = glob(os.path.join(dirname, 'LC*_MTL.txt'))
    assert len(files_mtl) == 1
    file_mtl = files_mtl[0]
    data_mtl = read_meta(file_mtl)['L1_METADATA_FILE']

    return data_mtl


def read_coordinates(ds, dirname, naming, chunksize):
    '''
    read lat/lon
    '''
    ds[naming.lat] = common.DataArray_from_array(
        LATLON(dirname, 'lat'),
        naming.dim2,
        chunksize,
    )
    ds[naming.lon] = common.DataArray_from_array(
        LATLON(dirname, 'lon'),
        naming.dim2,
        chunksize,
    )
    ds.attrs[naming.totalheight] = ds.rows.size
    ds.attrs[naming.totalwidth] = ds.columns.size


def gen_l8_angles(dirname, l8_angles=None):
    print(f'Geometry file is missing in {dirname}, generating it with {l8_angles}...')
    angles_txt_file = glob(os.path.join(dirname, 'LC08_*_ANG.txt'))
    assert len(angles_txt_file) == 1
    assert l8_angles is not None
    assert os.path.exists(l8_angles)
    path_exe = os.path.abspath(l8_angles)
    path_angles = os.path.abspath(angles_txt_file[0])
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = f'cd {tmpdir} ; {path_exe} {path_angles} BOTH 1 -b 1'
        os.system(cmd)
        angle_files = os.path.join(tmpdir, '*')
        os.system(f'cp -v {angle_files} {dirname}')


def read_geometry(ds, dirname, l8_angles, naming, chunksize):
    filenames_sensor = glob(os.path.join(dirname, 'LC*_sensor_B01.img'))

    if (not filenames_sensor) and (l8_angles is not None):
        gen_l8_angles(dirname, l8_angles)
        filenames_sensor = glob(os.path.join(dirname, 'LC*_sensor_B01.img'))

    # read sensor angles
    assert len(filenames_sensor) == 1, \
        f'Error, sensor angles file missing in {dirname} ({str(filenames_sensor)})'
    filename_sensor = filenames_sensor[0]
    data_sensor = da.from_array(
        np.memmap(filename_sensor,
                  dtype='int16',
                  mode='r',
                  order='C',
                  shape=(2, ds.totalheight, ds.totalwidth)),
        chunks=(1,)+chunksize,
        meta=np.array([], 'int16'),
        )

    ds[naming.vza] = (naming.dim2, data_sensor[1, :, :]/100.)
    ds[naming.vaa] = (naming.dim2, data_sensor[0, :, :]/100.)


    # read solar angles
    filenames_solar = glob(os.path.join(dirname, 'LC*_solar_B01.img'))
    assert len(filenames_solar) == 1, \
        'Error, solar angles file missing ({})'.format(str(filenames_solar))
    filename_solar = filenames_solar[0]
    data_solar = da.from_array(
        np.memmap(filename_solar,
                  dtype='int16',
                  mode='r',
                  order='C',
                  shape=(2, ds.totalheight, ds.totalwidth)),
        chunks=(1,)+chunksize,
        meta=np.array([], 'int16'),
        )
    ds[naming.sza] = (naming.dim2, data_solar[1, :, :]/100.)
    ds[naming.saa] = (naming.dim2, data_solar[0, :, :]/100.)


def read_radiometry(ds, dirname, split, data_mtl, radiometry, naming, chunksize):
    param = {'reflectance': naming.Rtoa,
             'radiance': naming.Ltoa}[radiometry]
    bnames = []
    for b in bands_oli:
        bname = (param+'_{}').format(b)
        bnames.append(bname)
        ds[bname] = common.DataArray_from_array(
            TOA_READ(b, dirname, radiometry, data_mtl),
            naming.dim2,
            chunksize,
        )
        if radiometry == 'reflectance':
            ds[bname] /= da.cos(da.radians(ds.sza))

    if not split:
        ds = eo.merge(ds, bnames,
                      param,
                      'bands',
                      coords=bands_oli)

    return ds


class LATLON:
    def __init__(self, dirname, kind):
        '''
        kind: 'lat' or 'lon'
        '''
        self.kind = kind

        files_B1 = glob(os.path.join(dirname, 'LC*_B1.TIF'))
        if len(files_B1) != 1:
            raise Exception('Invalid directory content ({})'.format(files_B1))
        file_B1 = files_B1[0]

        b1 = gdal.Open(file_B1)
        old_cs = osr.SpatialReference()
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
        self.dtype = np.dtype('float64')


    def __getitem__(self, keys):
        x = self.X[keys[1]]
        y = self.Y[keys[0]]
        XY = np.array(np.meshgrid(x, y))
        XY = np.moveaxis(XY, 0, -1)

        # get the coordinates in lat long
        latlon = np.array(self.transform.TransformPoints(XY.reshape((-1, 2))))
        assert latlon.dtype == self.dtype
        if self.kind == 'lat':
            return latlon[:, 1].reshape((len(y), len(x)))
        else:
            assert self.kind == 'lon'
            return latlon[:, 0].reshape((len(y), len(x)))

class TOA_READ:
    '''
    An array-like to read Landsat-8 OLI TOA reflectance
    (have to be divided by cos(sza) - radiance and reflectance
    in L1 are independent of the geometry)

    Arguments:
        b: band identifier (440, 480, 560, 655, 865)
        kind: 'radiance' or 'reflectance'
    '''
    def __init__(self, b, dirname, radiometry='reflectance', data_mtl=None):
        if data_mtl is None:
            data_mtl = read_metadata(dirname)

        if radiometry == 'reflectance':
            param_mult = 'REFLECTANCE_MULT_BAND_{}'
            param_add = 'REFLECTANCE_ADD_BAND_{}'
        elif radiometry == 'radiance':
            param_mult = 'RADIANCE_MULT_BAND_{}'
            param_add = 'RADIANCE_ADD_BAND_{}'
        else:
            raise Exception('TOA_READ: `kind` should be `radiance` or `reflectance`')

        self.M = data_mtl['RADIOMETRIC_RESCALING'][param_mult.format(band_index[b])]
        self.A = data_mtl['RADIOMETRIC_RESCALING'][param_add.format(band_index[b])]

        self.filename = os.path.join(
            dirname,
            data_mtl['PRODUCT_METADATA']['FILE_NAME_BAND_{}'.format(band_index[b])])

        assert os.path.exists(self.filename)
        dset = gdal.Open(self.filename)
        band = dset.GetRasterBand(1)
        self.dtype = np.dtype('float64')
        self.width = band.XSize
        self.height = band.YSize
        self.shape = (self.height, self.width)
        self.ndim = 2

    def __getitem__(self, keys):
        ystart = int(keys[0].start) if keys[0].start is not None else 0
        xstart = int(keys[1].start) if keys[1].start is not None else 0
        ystop = int(keys[0].stop) if keys[0].stop is not None else self.shape[0]
        xstop = int(keys[1].stop) if keys[1].stop is not None else self.shape[1]

        dset = gdal.Open(self.filename)   # NOTE: we have to re-open the file each time to avoid a segfault
        band = dset.GetRasterBand(1)
        data = band.ReadAsArray(  # NOTE: step is not supported by gdal, have to apply a posteriori
            xoff=xstart,
            yoff=ystart,
            win_xsize=xstop - xstart,
            win_ysize=ystop - ystart,
            )[::keys[0].step, ::keys[1].step]
        assert data is not None
        r = self.M*data + self.A
        assert r.dtype == self.dtype
        return r


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
            if '.' in value: #float
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

