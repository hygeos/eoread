#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for exploiting eoread objects
'''

import shutil
import tempfile
from pathlib import Path
from collections import OrderedDict
from contextlib import contextmanager
import re

import dask.array as da
from dask.diagnostics import ProgressBar
import xarray as xr
from dateutil.parser import parse
import numpy as np
from numpy import arcsin as asin
from numpy import cos, radians, sin, sqrt, where
from shapely.geometry import Point, Polygon

from .naming import naming


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

def scattering_angle(mu_s, mu_v, phi):
    """
    Scattering angle

    mu_s: cos of the sun zenith angle
    mu_v: cos of the view zenith angle
    phi: relative azimuth angle in degrees
    """
    return -mu_s*mu_v - sqrt((1.-mu_s*mu_s)*(1.-mu_v*mu_v)) * cos(radians(phi))


def init_geometry(ds, scat_angle=False):
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

    # relative azimuth angle
    if 'raa' not in ds:
        raa = ds.saa - ds.vaa
        raa = raa % 360
        ds['raa'] = raa.where(raa < 180, 360-raa)
        ds.raa.attrs['description'] = 'relative azimuth angle'
        ds.raa.attrs['unit'] = 'degrees'

    # scattering angle
    if scat_angle:
        ds['scat_angle'] = scattering_angle(ds.mus, ds.muv, ds.raa)
        ds['scat_angle'].attrs = 'scattering angle'

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

    drop_invalid, bool
        if True invalid pixels will be replace by nan for floats
        and int_default_value for other types

    int_default_value, int
        for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    lat = ds[naming.lat].compute()
    lon = ds[naming.lon].compute()
    cond = haversine(lat, lon, pt_lat, pt_lon) < rad
    cond = cond.compute()

    return sub(ds, cond, drop_invalid, int_default_value)


def none_context(a=None):
    """
    Returns a context manager that does nothing.

    In python 3.7, this is equivalent to `contextlib.nullcontext`.
    """
    return contextmanager(lambda: (x for x in [a]))()


def to_netcdf(ds, *,
              filename=None,
              dirname=None,
              product_name=None,
              ext='.nc',
              product_name_attr='product_name',
              overwrite=False,
              tmpdir=None,
              create_out_dir=True,
              verbose=True,
              **kwargs):
    '''
    Write an xarray Dataset `ds` using `.to_netcdf` with several additional features:
    - construct file name using  `dirname`, `product_name` and `ext`
      (optionnally - otherwise with `filename`)
    - check that the output file does not exist already
    - Use file compression
    - Use temporary file
    - Create output directory if it does not exist

    Arguments:
    ----------

    filename: str or None
        Output file path.
        if None, build filename from `dirname`, `product_name` and `ext`.
    dirname: str or None
        directory for output file (default None: uses the attribute input_directory of ds)
    product_name: str
        base name for the output file. If None (default), use the attribute named attr_name
    ext: str
        extension (default: '.nc')
    product_name_attr: str
        name of the attribute to use for product_name in `ds`
    overwrite: bool or 'skip'
        whether to overwrite existing file (default: False ; raises an error).
    tmpdir: str
        use a given temporary directory instead of the output directory
    create_out_dir: str
        create output directory if it does not exist

    Other kwargs are passed to `to_netcdf`

    About engine and compression:
        - Use default engine='h5netcdf' (much faster than 'netcdf4' when activating compression)
        - Use compression by default: encoding={'zlib':True, 'complevel':9}.
          Compression can be disactivated by passing encoding={}


    Returns:
    -------

    output file name (str)
    '''
    assert isinstance(ds, xr.Dataset), 'eo.to_netcdf expects an xarray Dataset'
    if filename is None:
        # construct filename from dirname, product_name and ext

        if product_name is None:
            product_name = ds.attrs[product_name_attr]
        assert product_name, 'Empty product name'
        if dirname is None:
            dirname = ds.attrs[naming.input_directory]
        fname = Path(dirname).resolve()/(product_name+ext)

    else:
        fname = Path(filename).resolve()

    if fname.exists():
        if overwrite == 'skip':
            print(f'File {fname} exists, skipping...')
            return
        elif overwrite:
            fname.unlink()
        else:
            raise IOError(f'Output file "{fname}" exists.')

    if not fname.parent.exists():
        if create_out_dir:
            fname.parent.mkdir(parents=True)
        else:
            raise IOError(f'Directory "{fname.parent}" does not exist.')

    defaults = {
        'engine': 'h5netcdf',
        'encoding': {var: dict(zlib=True, complevel=9)
                     for var in ds.data_vars}
    }
    defaults.update(kwargs)

    PBar = {
        True: ProgressBar,
        False: none_context
    }[verbose]

    with PBar(), tempfile.TemporaryDirectory(dir=tmpdir) as tmp:

        fname_tmp = Path(tmp)/fname.name

        if verbose:
            print('Writing:', fname)
            print('Using temporary file:', fname_tmp)

        ds.to_netcdf(path=fname_tmp,
                     **defaults)

        # use intermediary move
        # (both files may be on different devices)
        shutil.move(fname_tmp, str(fname)+'.tmp')
        shutil.move(str(fname)+'.tmp', fname)

    return fname


def split(d, dim, sep='_'):
    '''
    Returns a Dataset where a given dimension is split into as many variables

    d: Dataset or DataArray
    '''
    assert dim in d.dims
    assert dim in d.coords, f'The split dimension "{dim}" must have coordinates.'

    if isinstance(d, xr.DataArray):
        m = xr.merge([
            d.isel(**{dim: i}).rename(f'{d.name}{sep}{d[dim].data[i]}').drop(dim)
            for i in range(len(d[dim]))
            ])
    elif isinstance(d, xr.Dataset):
        m = xr.merge(
            [split(d[x], dim)
             if dim in d[x].dims
             else d[x]
             for x in d])
    else:
        raise Exception('`split` expects Dataset or DataArray.')

    m.attrs.update(d.attrs)
    m.attrs['split_dimension'] = dim
    m = m.assign_coords(**d.coords)
    return m


def merge(ds,
          dim=None,
          pattern=r'(.+)_(\d+)',
          dtype=int):
    """
    Merge DataArrays in `ds` along dimension `dim`.

    ds: xr.Dataset

    dim: str or None
        name of the new or existing dimension
        if None, use the attribute `split_dimension`

    pattern: str
        Regular expression for matching variable names - must consist of two groups.
        First group represents the new variable name.
        Second group represents the coordinate value
        Ex: r'(.+)_(\d+)'
            First group matches non-digit.
            Second group matches digits.

    dtype: data type
        data type of the coordinate items
    """
    copy = ds.copy()

    if dim is None:
        dim = copy.attrs['split_dimension']

    mapping = {}
    for x in copy:
        m = re.findall(pattern, x)
        if not m:
            continue  # does not match
        assert len(m) == 1, 'Expecting a single match'
        assert len(m[0]) == 2, 'Expecting two groups in regular expression'
        var, coord = m[0]
        c = dtype(coord)

        if var not in mapping:
            mapping[var] = []

        mapping[var].append((x, c))

    for var in mapping:
        data = xr.concat([copy[x] for x, c in mapping[var]], dim)
        coords = [c for x, c in mapping[var]]
        if dim in copy.coords:
            # check that the coordinates are matching
            existing_coords = list(copy.coords['bands'].data)
            assert existing_coords == coords
        else:
            copy = copy.assign_coords(**{dim: coords})
        copy[var] = data
        copy = copy.drop([x for x, c in mapping[var]])

    return copy


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


def getflags(A):
    """
    returns the flags in attributes of `A` as a dictionary

    Arguments:
    ---------

    A: Dataarray
    """
    try:
        m = A.attrs[naming.flags_meanings].split(naming.flags_meanings_separator)
        v = A.attrs[naming.flags_masks]
    except KeyError:
        return OrderedDict()
    return OrderedDict(zip(m, v))


def getflag(A, name):
    """
    Return the binary flag with given `name` as a boolean array

    A: DataArray
    name: str

    example: getflag(flags, 'LAND')
    """
    flags = getflags(A)

    assert name in flags, f'Error, {name} no in {list(flags)}'

    return (A & flags[name]) != 0 


def raiseflag(A, flag_name, flag_value, condition):
    """
    Raise a flag in DataArray `A` with name `flag_name`, value `flag_value` and `condition`
    The name and value of the flag is recorded in the attributes of `A`

    Arguments:
    ----------
    A: DataArray of integers

    flag_name: str
        Name of the flag
    flag_value: int
        Value of the flag
    condition: boolean array-like of same shape as `A`
        Condition to raise flag
    """
    flags = getflags(A)
    dtype_flag_masks = 'uint16'

    if naming.flags_meanings not in A.attrs:
        A.attrs[naming.flags_meanings] = ''
    if naming.flags_masks not in A.attrs:
        A.attrs[naming.flags_masks] = np.array([], dtype=dtype_flag_masks)

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

        A.attrs[naming.flags_meanings] = naming.flags_meanings_separator.join(keys)
        A.attrs[naming.flags_masks] = np.array(values, dtype=dtype_flag_masks)

    notraised = (A & flag_value) == 0
    A += flag_value * (condition & notraised).astype(naming.flags_dtype)
