#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xarray as xr
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon, Point

from numpy import radians, cos, sin, arcsin as asin, sqrt

def haversine(lat1, lon1, lat2, lon2, radius=6371):
    '''
    Calculate the great circle distance between two points (specified in
    decimal degrees) on a sphere of a given radius

    Returns the distance in the same unit as radius (defaults to earth radius in km)
    '''
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = [radians(x) for x in [lon1, lat1, lon2, lat2]]

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    dist = radius * c

    return dist

@xr.register_dataset_accessor('eo')
class GeoDatasetAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def init(self):
        '''
        Product initialization: lazily add useful variables
        '''
        ds = self._obj
        print(f'Initialize product {ds.product_name}...')

        # set rows and column indices
        # this is necessary to keey the position of a subproduct
        # within a product
        for x in ['rows', 'columns']:
            if len(ds[x].coords) == 0:
                ds[x] = xr.IndexVariable(x, range(len(ds[x])))

        # mus and muv
        assert not 'mus' in ds
        assert not 'muv' in ds
        ds['mus'] = np.cos(np.radians(ds.sza))
        ds['mus'].attrs['description'] = 'cosine of the sun zenith angle'
        ds['muv'] = np.cos(np.radians(ds.vza))
        ds['muv'].attrs['description'] = 'cosine of the view zenith angle'

        # TODO: scattering angle

        # TOA reflectance
        if 'Rtoa' not in ds:
            ds['Rtoa'] = np.pi*ds.Ltoa/(ds.mus*ds.F0)

        return ds
    
    def locate(self, lat, lon):
        print(f'Locating lat={lat}, lon={lon}')
        ds = self._obj
        # TODO: haversine
        dist = (ds.latitude - lat)**2 + (ds.longitude - lon) **2
        dist_min = np.amin(dist)
        # TODO: check if it is within
        return np.where(dist == dist_min)

    def contains(self, lat, lon):
        pt = Point(lat,lon)
        area = Polygon(self._obj.attrs['Footprint'])
        # TODO: proper inclusion test
        # TODO: make it work with arrays
        return area.contains(pt)

    def check(self):
        datasets = ['Ltoa', 'sza', 'vza', 'vaa', 'saa']
        for x in datasets:
            assert 'units' in self._obj[x].attrs, f'{x} has no units'


    def show_footprint(self):
        import ipyleaflet as ipy

        poly_pts = self._obj.attrs['Footprint']
        center = [sum(x)/len(poly_pts) for x in zip(*poly_pts)]

        map = ipy.Map(zoom=4, center = center)
        polygon = ipy.Polygon(locations = poly_pts, color = "green", fillcolor = "blue")
        map.add_layer(polygon)
        
        return map
    
    def sub(self, cond):
        '''
        Creates a Dataset based on the conditions passed in parameters

        cond : must be a DataArray of booleans
        '''
        res = xr.Dataset()
        ds = self._obj

        for var in ds.variables:
            if set(cond.dims) == set(ds[var].dims).intersection(set(cond.dims)):
                res[var] = ds[var].where(cond, drop = True)
            else:
                res[var] = ds[var]

        res.attrs.update(ds.attrs)

        return res


    def sub_rect(self, lat_min, lat_max, lon_min, lon_max):
        '''
        Creates a Dataset based on the coordonates of the rectangle passed in parameters

        lat_min, lat_max, lon_min, lon_max : delimitations of the zone desired
        '''
        lat = self._obj.latitude.compute()
        lon = self._obj.longitude.compute()
        cond = (lat < lat_max) & (lat > lat_min) & (lon < lon_max) & (lon > lon_min)
        cond = cond.compute()

        #fname = '_Rect_' + str(lat_min) + '_' + str(lat_max) + '|' + str(lon_min) + '_' + str(lon_max)

        return self.sub(cond)

    def sub_pt(self, pt_lat, pt_lon, rad):
        '''
        Creates a Dataset based on the circle specified in parameters

        pt_lat, pt_lon : Coordonates of the center of the point

        rad : radius of the circle in km
        '''
        lat = self._obj.latitude.compute()
        lon = self._obj.longitude.compute()
        cond = haversine(lat, lon, pt_lat, pt_lon) < rad
        cond = cond.compute()

        return self.sub(cond)
    
    def to_netcdf(self, dirname='.', suffix='', attr_name='product_name', compress=True, **kwargs):
        '''
        Write a xr.Dataset using product_name attribute and a suffix
        '''
        suffix = suffix+'.nc'
        fname = os.path.join(dirname, self._obj.attrs[attr_name]+suffix)
        fname_tmp = fname+'.tmp'

        encoding = {}
        if compress:
            comp = dict(zlib=True, complevel=1)
            encoding = {var: comp for var in self._obj.data_vars}
        
        self._obj.to_netcdf(path=fname_tmp, encoding = encoding, **kwargs)
        os.rename(fname_tmp, fname)

    def split(self, var_name, out_var=None, split_axis='bands'):
        copy = self._obj.copy(deep=True)
        if not out_var:
            out_var = var_name+'_'
        if not (split_axis in copy[var_name]):
            raise Exception("variable '{}' doesn't have '{}' dimension".format(var_name, split_axis))
        for x in copy[var_name][split_axis]:
            copy[out_var+str(x.data)] = copy[var_name].sel({split_axis : x})
        # delete split dimension
        
        return copy

    def merge(self, var_names, out_var, new_dim):
        pass

@xr.register_dataarray_accessor('eo')
class GeoDataArrayAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def view(self):
        raise NotImplementedError


class AtIndex(object):
    '''
    Use DataArray idx to index DataArray A along dimension idx_name

    Example:
        A: DataArray (nbands x detectors)
        idx: DataArray (rows x columns)
        Results in A[idx]: (nbands x rows x columns)
    '''
    def __init__(self, A, idx, idx_name):
        # dimensions to be indexed by this object
        self.dims = sum([[x] if not x == idx_name else list(idx.dims) for x in A.dims], [])
        # ... and their shape
        shape = sum([[A.shape[i]] if not x == idx_name else list(idx.shape) for i, x in enumerate(A.dims)], [])
        self.shape = shape
        self.dtype = A.dtype
        self.A = A
        self.idx = idx
        self.pos_dims_idx = [i for i, x in enumerate(self.dims) if x in idx.dims]
        self.idx_name = idx_name

    def __getitem__(self, key):
        # first index idx using the appropriate dimensions in key
        idx = self.idx[tuple([key[i] for i in self.pos_dims_idx])].values

        # then index A using the remaining dimensions
        return self.A.values[tuple([key[i] if k != self.idx_name else idx
                             for i, k in enumerate(self.A.dims)])]


class Interpolator(object):
    def __init__(self, shape, A):
        self.shape = shape
        self.dtype = A.dtype
        self.A = A
        assert A.dims == ('tie_rows', 'tie_columns')

    def __getitem__(self, key):
        ret =  self.A.interp(
                tie_rows=np.arange(self.shape[0])[key[0]],
                tie_columns=np.arange(self.shape[1])[key[1]],
                )
        return ret

class Repeat(object):
    def __init__(self, A, repeats):
        '''
        Repeat elements of A
            repeats: tuple (number of repeats along each dimension)
        '''
        self.shape = tuple([s*r for (s, r) in zip(A.shape, repeats)])
        self.dtype = A.dtype
        self.A = A
    
    def __getitem__(self, key):
        return 


def rectBivariateSpline(A, shp):
    '''
    Bivariate spline interpolation of array A to shape shp.

    Fill NaNs with closest values, otherwise RectBivariateSpline gives no
    result.
    '''
    xin = np.arange(shp[0], dtype='float32') / (shp[0]-1) * A.shape[0]
    yin = np.arange(shp[1], dtype='float32') / (shp[1]-1) * A.shape[1]

    x = np.arange(A.shape[0], dtype='float32')
    y = np.arange(A.shape[1], dtype='float32')

    invalid = np.isnan(A)
    if invalid.any():
        # fill nans
        # see http://stackoverflow.com/questions/3662361/
        ind = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        A = A[tuple(ind)]

    f = RectBivariateSpline(x, y, A)

    return f(xin, yin).astype('float32')
