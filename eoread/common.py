#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon, Point

from numpy import radians, cos, sin, arcsin as asin, sqrt, where

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

    def init(self, optional=[]):
        '''
        Common initializations + optional ones
        optional: list of variables to initialize:
            'Rtoa': TOA reflectance (level1 products)
                    -> implies 'geometry'
            'geometry': geometrical variables
        '''
        ds = self._obj

        # set rows and column indices
        # this is necessary to keep the position of a subproduct
        # within a product
        for x in ['rows', 'columns']:
            if len(ds[x].coords) == 0:
                ds[x] = xr.IndexVariable(x, range(len(ds[x])))

        if 'Rtoa' in optional:
            self.init_Rtoa()
        if 'geometry' in optional:
            self.init_geometry()

        return ds

    def init_Rtoa(self):
        '''
        Initialize TOA reflectances
        '''
        ds = self._obj

        self.init_geometry()

        # TOA reflectance
        if 'Rtoa' not in ds:
            ds['Rtoa'] = np.pi*ds.Ltoa/(ds.mus*ds.F0)

        return ds

    def init_geometry(self):
        '''
        Initialize geometry variables
        '''
        ds = self._obj

        # mus and muv
        if 'mus' not in ds:
            ds['mus'] = np.cos(np.radians(ds.sza))
            ds['mus'].attrs['description'] = 'cosine of the sun zenith angle'
        if 'muv' not in ds:
            ds['muv'] = np.cos(np.radians(ds.vza))
            ds['muv'].attrs['description'] = 'cosine of the view zenith angle'

        # TODO: scattering angle

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
    
    def sub(self, cond, drop_invalid=True, int_default_value=0):
        '''
        Creates a Dataset based on the conditions passed in parameters

        cond : a DataArray of booleans that defines which pixels are kept

        drop_invalid, bool : if True invalid pixels will be replace by nan for floats and int_default_value for other types

        int_default_value, int : for DataArrays of type int, this value is assigned on non-valid pixels
        '''
        res = xr.Dataset()
        ds = self._obj

        if drop_invalid:
            assert 'mask_valid' not in res
            res['mask_valid'] = cond.where(cond, drop=True)
            res['mask_valid'] = res['mask_valid'].where(~np.isnan(res['mask_valid']), 0).astype(bool)

        slice_dict = dict()
        for dim in cond.dims:
            s = cond.any(dim=[d for d in cond.dims if d != dim])
            wh = where(s)[0]
            if len(wh)==0:
                slice_dict[dim]=slice(2,1)
            else:
                slice_dict[dim]=slice(wh[0], wh[-1]+1)

        for var in ds.variables:
            if set(cond.dims) == set(ds[var].dims).intersection(set(cond.dims)):
                if drop_invalid:
                    if ds[var].dtype in ['float16', 'float32', 'float64']:
                        res[var] = ds[var].where(cond, drop=True)
                    else:
                        res[var] = ds[var].isel(slice_dict).where(res['mask_valid'], int_default_value)

                else:
                    res[var] = ds[var].isel(slice_dict)

        res.attrs.update(ds.attrs)

        return res

    def sub_rect(self, lon_min, lat_min, lon_max, lat_max, drop_invalid=True, int_default_value=0):
        '''
        Creates a Dataset based on the coordonates of the rectangle passed in parameters

        lat_min, lat_max, lon_min, lon_max : delimitations of the zone desired

        drop_invalid, bool : if True invalid pixels will be replace by nan for floats and int_default_value for other types

        int_default_value, int : for DataArrays of type int, this value is assigned on non-valid pixels
        '''
        lat = self._obj.latitude.compute()
        lon = self._obj.longitude.compute()
        cond = (lat < lat_max) & (lat > lat_min) & (lon < lon_max) & (lon > lon_min)
        cond = cond.compute()

        return self.sub(cond, drop_invalid, int_default_value)

    def sub_pt(self, pt_lat, pt_lon, rad, drop_invalid=True, int_default_value=0):
        '''
        Creates a Dataset based on the circle specified in parameters

        pt_lat, pt_lon : Coordonates of the center of the point

        rad : radius of the circle in km

        drop_invalid, bool : if True invalid pixels will be replace by nan for floats and int_default_value for other types

        int_default_value, int : for DataArrays of type int, this value is assigned on non-valid pixels
        '''
        lat = self._obj.latitude.compute()
        lon = self._obj.longitude.compute()
        cond = haversine(lat, lon, pt_lat, pt_lon) < rad
        cond = cond.compute()

        return self.sub(cond, drop_invalid, int_default_value)

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

    def split(self, var_name, out_vars=None, split_axis=None, drop=True):
        """
        Returns a DataSet where the variable 'var_name' is split into many variables along the 'split_axis' dimension.

        var_name, str : name of the variable to split
        out_vars, str or list of str : names or prefix of the output variables concatenated with their value in the 'split_axis' axis
                       by default, it uses the var_name as prefix
        split_axis, str : name of the axis along which the variable is split

        drop : bool, if True, variable var_name is deleted in the returned DataSet
        """
        copy = self._obj.copy()
        if not split_axis:
            split_axis = copy[var_name].dims[0]
        if not (split_axis in copy[var_name].dims):
            raise Exception("variable '{}' doesn't have '{}' dimension".format(var_name, split_axis))

        if isinstance(out_vars, list):
            cpt=0
            for x in copy[var_name][split_axis]:
                copy[out_vars[cpt]] = copy[var_name].sel({split_axis : x})
                cpt+=1
        else:
            if not out_vars:
                out_vars = var_name+'_'
            for x in copy[var_name][split_axis]:
                copy[out_vars+str(x.data)] = copy[var_name].sel({split_axis : x})
        
        if drop:
            copy = copy.drop(var_name)
        return copy

    def merge(self, var_names, out_var, new_dim_name, coords=None,
              dim_index=0, drop=True):
        """
        Returns a DataSet where all the variables included in the 'var_names' list are merged into a
        new variable named 'out_var'.
        If the 'new_dim' dimension already exists, the variables are concatenated along the dimension,
        otherwise it creates this dimension in the new variable

        var_names, list of str : names of variables to concatenate
        out_var, str : the output variable name created
        new_dim_name, str : name of the dimension along the variables are concatenated
        coords: coordinates along  the new dimension
        dim_index, int: index where to put the new dimension
        drop : bool, if True, variables in var_names are deleted in the returned DataSet
        """
        copy = self._obj.copy()
        if out_var in list(copy.variables):
            raise Exception("variable '{}' already exists in the dataset".format(out_var))
        
        data = xr.concat([copy[var] for var in var_names], new_dim_name)

        dims = [dim for dim in copy[var_names[0]].dims]
        if dim_index < 0:
            dim_index = len(dims)+1+dim_index
        dims.insert(dim_index, new_dim_name)
        data = data.transpose(*dims)
        if coords is not None:
            data = data.assign_coords(**{new_dim_name: coords})

        if drop:
            copy = copy.drop([var for var in var_names])

        return copy.assign({out_var: data})

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
    '''
    An array-like object to interpolate 2-dim array `A` to new `shape`

    Uses coordinates `tie_rows` and `tie_columns`.
    '''
    def __init__(self, shape, A):
        self.shape = shape
        self.dtype = A.dtype
        self.A = A
        assert A.dims == ('tie_rows', 'tie_columns')
        self.ndim = 2

    def __getitem__(self, key):
        ret = self.A.interp(
            tie_rows=np.arange(self.shape[0])[key[0]],
            tie_columns=np.arange(self.shape[1])[key[1]],
        )
        return ret

class Repeat(object):
    def __init__(self, A, repeats):
        '''
        Repeat elements of `A`

        Parameters:
        A: DataArray to repeat
        repeats: tuple of int (number of repeats along each dimension)
        '''
        self.shape = tuple([s*r for (s, r) in zip(A.shape, repeats)])
        self.ndim = len(self.shape)
        self.repeats = repeats
        self.dtype = A.dtype
        self.A = A

    def __getitem__(self, key):
        indices = [np.arange(self.shape[i], dtype='int')[k]//self.repeats[i]
                   for i, k in enumerate(key)]
        X, Y = np.meshgrid(*indices)
        return np.array(self.A)[X, Y].transpose()


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
