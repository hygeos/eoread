#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for exploiting eoread objects
'''

import os
from dateutil.parser import parse
from numpy import radians, cos, sin, arcsin as asin, sqrt, where
import numpy as np
import xarray as xr
import tempfile
import dask.array as da
from dask.diagnostics import ProgressBar
from shapely.geometry import Polygon, Point
from .naming import naming
import shutil
from collections import OrderedDict


def datetime(ds):
    '''
    Parse datetime (in isoformat) from `ds` attributes
    '''
    return parse(ds.datetime)


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


def init_Rtoa(ds):
    '''
    Initialize TOA reflectances from radiance (in place)

    Implies init_geometry
    '''
    init_geometry(ds)

    # TOA reflectance
    if naming.Rtoa not in ds:
        ds[naming.Rtoa] = np.pi*ds[naming.Ltoa]/(ds.mus*ds[naming.F0])

    return ds

def init_geometry(ds):
    '''
    Initialize geometric variables (in place)
    '''

    # mus and muv
    if 'mus' not in ds:
        ds['mus'] = np.cos(np.radians(ds.sza))
        ds['mus'].attrs['description'] = 'cosine of the sun zenith angle'
    if 'muv' not in ds:
        ds['muv'] = np.cos(np.radians(ds.vza))
        ds['muv'].attrs['description'] = 'cosine of the view zenith angle'

    # TODO: scattering angle

    # relative azimuth angle
    if 'raa' not in ds:
        raa = ds.saa - ds.vaa
        raa = raa % 360
        ds['raa'] = raa.where(raa < 180, 360-raa)
        ds.raa.attrs['description'] = 'relative azimuth angle'
        ds.raa.attrs['unit'] = 'degrees'

    return ds


def locate(ds, lat, lon):
    print(f'Locating lat={lat}, lon={lon}')
    # TODO: haversine
    dist = (ds.latitude - lat)**2 + (ds.longitude - lon) **2
    dist_min = np.amin(dist)
    # TODO: check if it is within
    return np.where(dist == dist_min)


def contains(ds, lat, lon):
    pt = Point(lat, lon)
    area = Polygon(zip(
        ds.attrs[naming.footprint_lat],
        ds.attrs[naming.footprint_lon]
    ))
    # TODO: proper inclusion test
    # TODO: make it work with arrays
    return area.contains(pt)


def show_footprint(ds, zoom=4):
    import ipyleaflet as ipy

    poly_pts = ds.attrs['Footprint']
    center = [sum(x)/len(poly_pts) for x in zip(*poly_pts)]

    m = ipy.Map(zoom=zoom,
                center=center)
    polygon = ipy.Polygon(locations=poly_pts,
                          color="green",
                          fillcolor="blue")
    m.add_layer(polygon)
    
    return m


def sub(ds, cond, drop_invalid=True, int_default_value=0):
    '''
    Creates a Dataset based on the conditions passed in parameters

    cond : a DataArray of booleans that defines which pixels are kept

    drop_invalid, bool : if True invalid pixels will be replace by nan for floats and int_default_value for other types

    int_default_value, int : for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    res = xr.Dataset()

    if drop_invalid:
        assert 'mask_valid' not in res
        res['mask_valid'] = cond.where(cond, drop=True)
        res['mask_valid'] = res['mask_valid'].where(~np.isnan(res['mask_valid']), 0).astype(bool)

    slice_dict = dict()
    for dim in cond.dims:
        s = cond.any(dim=[d for d in cond.dims if d != dim])
        wh = where(s)[0]
        if len(wh) == 0:
            slice_dict[dim] = slice(2,1)
        else:
            slice_dict[dim] = slice(wh[0], wh[-1]+1)

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


def sub_rect(ds, lat_min, lon_min, lat_max, lon_max, drop_invalid=True, int_default_value=0):
    '''
    Returns a Dataset based on the coordinates of the rectangle passed in parameters

    lat_min, lat_max, lon_min, lon_max : delimitations of the region of interest

    drop_invalid, bool : if True, invalid pixels will be replace by nan
    for floats and int_default_value for other types

    int_default_value, int : for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    lat = ds.latitude.compute()
    lon = ds.longitude.compute()
    cond = (lat < lat_max) & (lat > lat_min) & (lon < lon_max) & (lon > lon_min)
    cond = cond.compute()

    return sub(ds, cond, drop_invalid, int_default_value)


def sub_pt(ds, pt_lat, pt_lon, rad, drop_invalid=True, int_default_value=0):
    '''
    Creates a Dataset based on the circle specified in parameters

    pt_lat, pt_lon : Coordonates of the center of the point

    rad : radius of the circle in km

    drop_invalid, bool : if True invalid pixels will be replace by nan for floats and int_default_value for other types

    int_default_value, int : for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    lat = ds[naming.lat].compute()
    lon = ds[naming.lon].compute()
    cond = haversine(lat, lon, pt_lat, pt_lon) < rad
    cond = cond.compute()

    return sub(ds, cond, drop_invalid, int_default_value)


def to_netcdf(ds,
              dirname='.',
              product_name=None,
              ext='.nc',
              product_name_attr='product_name',
              overwrite=False,
              compress=True,
              tmpdir=None,
              create_out_dir=True,
              verbose=True,
              **kwargs):
    '''
    Write a xarray Dataset `ds` using `.to_netcdf` with several additional features:
    - construct file name using  `dirname`, `product_name` and `ext`
    - check that the output file does not exist already
    - Use file compression
    - Use temporary file
    - Create output directory if it does not exist

    Arguments:
    - dirname: directory for output file (default '.')
    - product_name: base name for the output file. If None (default), use the attribute named attr_name
    - product_name_attr: name of the attribute to use for product_name in `ds`
    - ext: extension (default: '.nc')
    - overwrite: whether to overwrite existing file (default: False ; raises an error).
    - compress: activate output file compression
    - tmpdir: use a given temporary directory instead of the output directory
    - create_out_dir: create output directory if it does not exist

    Other kwargs are passed to `to_netcdf`

    Returns: output file name
    '''
    if product_name is None:
        product_name = ds.attrs[product_name_attr]
    assert product_name, 'Empty product name'
    fname = os.path.join(dirname, product_name+ext)

    encoding = {}
    if compress:
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in ds.data_vars}

    if os.path.exists(fname):
        if overwrite:
            os.remove(fname)
        else:
            raise IOError(f'Output file "{fname}" exists.')

    if not os.path.exists(dirname):
        if create_out_dir:
            os.makedirs(dirname)
        else:
            raise IOError(f'Directory "{dirname}" does not exist.')

    with tempfile.TemporaryDirectory(dir=tmpdir) as tmp:

        fname_tmp = os.path.join(tmp, product_name+ext+'.tmp')

        if verbose:
            print('Writing:', fname)
            print('Using temporary file:', fname_tmp)
        with open(os.devnull, 'w') as devnull, \
             ProgressBar(out={True: None,
                              False: devnull}[verbose]):
            ds.to_netcdf(path=fname_tmp,
                         encoding=encoding,
                         **kwargs)

        # use intermediary move
        # (both files may be on different devices)
        shutil.move(fname_tmp, fname+'.tmp')
        shutil.move(fname+'.tmp', fname)

    return fname


def split(d, dim, sep=''):
    '''
    Returns a Dataset where a given dimension is split into as many variables

    d: Dataset or DataArray
    '''
    assert dim in d.dims

    if isinstance(d, xr.DataArray):
        return xr.merge([
            d.isel(**{dim: i}).rename(f'{d.name}{sep}{d[dim].data[i]}').drop(dim)
            for i in range(len(d[dim]))
            ])
    elif isinstance(d, xr.Dataset):
        merged = xr.merge(
            [split(d[x], dim)
             if dim in d[x].dims
             else d[x]
             for x in d])
        merged.attrs = d.attrs
        return merged
    else:
        raise Exception('`split` expects Dataset or DataArray.')

    
def merge(ds, var_names, out_var, new_dim_name, coords=None,
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
    copy = ds.copy()
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


def broadcast(A, B):
    """
    Broadcast DataArray `A` to match the dimensions of DataArray `B`

    Returns: the broadcasted DataArray
    """
    new_shp1 = tuple([
        s
        if d in A.dims
        else 1
        for (s, d) in zip(B.shape, B.dims)
        ])

    AA = A.data.reshape(new_shp1)
    for i, s in [(i, s)
                 for (i, (s, d)) in enumerate(zip(B.shape, B.dims))
                 if d not in A.dims
                 ]:
        AA = da.repeat(AA, s, axis=i)

    return xr.DataArray(
        AA,
        dims=B.dims,
    )


def getflags(da):
    """
    returns the flags in attributes of `da` as a dictionary

    Arguments:
    ---------

    da: Dataarray
    """
    try:
        m = da.attrs[naming.flags_meanings].split(naming.flags_meanings_separator)
        v = da.attrs[naming.flags_masks]
    except KeyError:
        return OrderedDict()
    return OrderedDict(zip(m, v))


def getflag(da, name):
    """
    Return the binary flag with given `name` as a boolean array

    example: getflag(flags, 'LAND')
    """
    flags = getflags(da)

    assert name in flags, f'Error, {name} no in {list(flags)}'

    return (da & flags[name]) != 0 


def raiseflag(da, flag_name, flag_value, condition):
    """
    Raise a flag in Dataarray `da` with name `flag_name`, value `flag_value` and `condition`
    The name and value of the flag is recorded in the attributes of `da`

    Arguments:
    ----------
    da: Dataarray of integers
    flag_name: str
        Name of the flag
    flag_value: int
        Value of the flag
    condition: boolean array-like of same shape as `da`
        Condition to raise flag
    """
    flags = getflags(da)
    dtype_flag_masks = 'uint16'

    if naming.flags_meanings not in da.attrs:
        da.attrs[naming.flags_meanings] = ''
    if naming.flags_masks not in da.attrs:
        da.attrs[naming.flags_masks] = np.array([], dtype=dtype_flag_masks)

    # update the attributes if necessary
    if flag_name in flags:
        # existing flag: check value
        assert flags[flag_name] == flag_value, \
            f'Flag {flag_name} already exists with a different value'
    else:
        assert flag_value not in flags.values(), \
            f'Flag value {flag_value} is already assigned to a different flags'

        flags[flag_name] = flag_value

        # sort the flags by values
        keys, values = zip(*sorted(flags.items(), key=lambda y: y[1]))

        da.attrs[naming.flags_meanings] = naming.flags_meanings_separator.join(keys)
        da.attrs[naming.flags_masks] = np.array(values, dtype=dtype_flag_masks)

    notraised = (da & flag_value) == 0
    da += flag_value * (condition & notraised).astype(naming.flags_dtype)

