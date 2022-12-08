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


def datetime(ds, attribute='datetime'):
    '''
    Parse datetime (in isoformat) from `ds` attributes
    '''
    return parse(ds.attrs[attribute]).replace(tzinfo=None)


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
    Scattering angle in degrees

    mu_s: cos of the sun zenith angle
    mu_v: cos of the view zenith angle
    phi: relative azimuth angle in degrees
    """
    sa = -mu_s*mu_v - sqrt((1.-mu_s*mu_s)*(1.-mu_v*mu_v)) * cos(radians(phi))
    return np.arccos(sa)*180./np.pi


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
        ds['scat_angle'].attrs['description'] = 'scattering angle'

    return ds


def locate(lat, lon, lat0, lon0, dist_min_km=None):
    """
    Locate `lat0`, `lon0` within `lat`, `lon`

    if dist_min_km is specified and if the minimal distance exceeds it, a ValueError is raised
    """
    print(f'Locating lat={lat0}, lon={lon0}')
    dist = haversine(lat, lon, lat0, lon0)
    dist_min = np.array(np.amin(dist))

    if (dist_min_km is not None) and (dist_min > dist_min_km):
        raise ValueError(f'locate: minimal distance is {dist_min}, should be at most {dist_min_km}')

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

    drop_invalid, bool
        if True invalid pixels will be replace by nan for floats and
        int_default_value for other types

    int_default_value, int
        for DataArrays of type int, this value is assigned on non-valid pixels
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
              if_exists='error',
              tmpdir=None,
              create_out_dir=True,
              engine='h5netcdf',
              zlib=True,
              complevel=5,
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
    if_exists: 'error', 'skip' or 'overwrite'
        what to do if output file exists
    tmpdir: str ; default None = system directory
        use a given temporary directory
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
        if if_exists == 'skip':
            print(f'File {fname} exists, skipping...')
            return fname
        elif if_exists == 'overwrite':
            fname.unlink()
        elif if_exists == 'error':
            raise IOError(f'Output file "{fname}" exists.')
        else:
            raise ValueError(f'Invalid option for "if_exists": {if_exists}')

    if not fname.parent.exists():
        if create_out_dir:
            fname.parent.mkdir(parents=True)
        else:
            raise IOError(f'Directory "{fname.parent}" does not exist.')

    encoding = {var: dict(zlib=True, complevel=complevel)
                for var in ds.data_vars} if zlib else None

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
                     engine=engine,
                     encoding=encoding,
                     **kwargs)

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
            d.isel(**{dim: i}).rename(f'{d.name}{sep}{d[dim].data[i]}').drop_vars(dim)
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
          varname=None,
          pattern=r'(.+)_(\d+)',
          dtype=int):
    r"""
    Merge DataArrays in `ds` along dimension `dim`.

    ds: xr.Dataset

    dim: str or None
        name of the new or existing dimension
        if None, use the attribute `split_dimension`
    
    varname: str or None
        name of the variable to create
        if None, detect variable name from regular expression

    pattern: str
        Regular expression for matching variable names and coordinates
        if varname is None:
            First group represents the new variable name.
            Second group represents the coordinate value
            Ex: r'(.+)_(\d+)'
                    First group matches all characters.
                    Second group matches digits.
                r'(\D+)(\d+)'
                    First group matches non-digit.
                    Second group matches digits.
        if varname is not None:
            Match a single group representing the coordinate value

    dtype: data type
        data type of the coordinate items
    """
    copy = ds.copy()

    if dim is None:
        dim = copy.attrs['split_dimension']

    mapping = {}   # {new_name: [(old_name, value), ...], ...}
    for x in copy:
        m = re.findall(pattern, x)
        if not m:
            continue  # does not match
        assert len(m) == 1, 'Expecting a single match'

        if varname is None:
            assert len(m[0]) == 2, 'Expecting two groups in regular expression'
            var, coord = m[0]
        else:
            assert not isinstance(m[0], tuple), 'Expecting a single group in regular expression'
            coord = m[0]
            var = varname
        c = dtype(coord)

        if var not in mapping:
            mapping[var] = []

        mapping[var].append((x, c))

    for var in mapping:
        data = xr.concat([copy[x] for x, c in mapping[var]], dim)
        coords = [c for x, c in mapping[var]]
        if dim in copy.coords:
            # check that the coordinates are matching
            existing_coords = list(copy.coords[dim].data)
            assert existing_coords == coords, \
                f'Error: {existing_coords} != {coords} (in variable {var})'
        else:
            copy = copy.assign_coords(**{dim: coords})
        copy[var] = data
        copy = copy.drop_vars([x for x, c in mapping[var]])

    return copy


def getflags(A=None, meanings=None, masks=None, sep=None):
    """
    returns the flags in attributes of `A` as a dictionary {meaning: value}

    Arguments:
    ---------

    provide either:
        A: Dataarray
    or:
        meanings: flag meanings 'FLAG1 FLAG2'
        masks: flag values [1, 2]
        sep: string separator
    """
    try:
        meanings = meanings if (meanings is not None) else A.attrs[naming.flags_meanings]
        masks = masks if (masks is not None) else A.attrs[naming.flags_masks]
        sep = sep or naming.flags_meanings_separator
    except KeyError:
        return OrderedDict()
    return OrderedDict(zip(meanings.split(sep), masks))


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
            f'Flag value {flag_value} is already assigned to a different flags (assigned flags are {flags.values()})'

        flags[flag_name] = flag_value

        # sort the flags by values
        keys, values = zip(*sorted(flags.items(), key=lambda y: y[1]))

        A.attrs[naming.flags_meanings] = naming.flags_meanings_separator.join(keys)
        A.attrs[naming.flags_masks] = np.array(values, dtype=dtype_flag_masks)

    notraised = (A & flag_value) == 0
    A += flag_value * ((condition != 0) & notraised).astype(naming.flags_dtype)


def wrap(ds, dim, vmin, vmax):
    """
    Wrap and reorder a cyclic dimension between vmin and vmax.
    The border value is duplicated at the edges.
    The period is (vmax-vmin)

    Example:
    * Dimension [0, 359] -> [-180, 180]
    * Dimension [-180, 179] -> [-180, 180]
    * Dimension [0, 359] -> [0, 360]

    Arguments:
    ----------

    ds: xarray.Dataset
    dim: str
        Name of the dimension to wrap
    vmin, vmax: float
        new values for the edges
    """

    pivot = vmax if (vmin < ds[dim][0]) else vmin

    left = ds.sel({dim: slice(None, pivot)})
    right = ds.sel({dim: slice(pivot, None)})

    if right[dim][-1] > vmax:
        # apply the offset at the right part
        right = right.assign_coords({dim: right[dim] - (vmax-vmin)})
    else:
        # apply the offset at the left part
        left = left.assign_coords({dim: left[dim] + (vmax-vmin)})

    # swaps the two parts
    return xr.concat([right, left], dim=dim)


def convert(A, unit_to, unit_from=None, converter=None):
    """
    Unit conversion

    Arguments:
    ---------

    A: DataArray to convert

    unit_from: str or None
        unit to convert from. If not provided, uses da.units

    unit_to: str
        unit to convert to
    
    converter: a dictionary for unit conversion
        example: converter={'Pa': 1, 'hPa': 1e-2}
    """
    if unit_from is None:
        unit_from = A.units
    
    default_converters = [
        # pressure
        {'Pa': 1,
         'hPa': 1e-2,
         'millibars': 1e-2,
         },

        # ozone
        {'kg/m2': 1,
         'kg m**-2': 1,
         'DU': 1/2.1415E-05,
         'Dobson units': 1/2.1415E-05,
         }
    ]

    conversion_factor = None
    for c in (default_converters if converter is None else [converter]):
        if (unit_from in c) and (unit_to in c):
            conversion_factor = c[unit_to]/c[unit_from]
            break

    if conversion_factor is None:
        raise ValueError(f'Unknown conversion from {unit_from} to {unit_to}')

    converted = A*conversion_factor
    converted.attrs['units'] = unit_to
    return converted


def chunk(ds, **kwargs):
    """
    Apply rechunking to a xr.Dataset `ds` along dimensions provided as kwargs

    Works like `ds.chunk` but works also for Datasets with repeated dimensions.
    """

    for var in ds:
        chks = [kwargs[d] if d in kwargs else None for d in ds[var].dims]
        if hasattr(ds[var].data, 'chunks') and len([c for c in chks if c is not None]):
            ds[var].data = ds[var].data.rechunk(chks)
            
    return ds


def trim_dims(A):
    """
    Trims the dimensions of Dataset A
    
    Rename all possible dimensions to avoid duplicate dimensions with same sizes
    Avoid any DataArray with duplicate dimensions
    """
    # list of lists of dimensions that should be grouped together
    groups = []
    
    # loop over all dimensions sizes
    for size in set(A.dims.values()):
        # list all dimensions with current size
        groups_current = []
        dims_current = [k for k, v in A.dims.items()
                        if v == size]

        # for each variable, add its dimensions (intersecting dims_current)
        # to separate groups to avoid duplicated
        for var in A:
            for i, d in enumerate(
                [x for x in A[var].dims
                 if x in dims_current]
                ):
                if len(groups_current) <= i:
                    groups_current.append([])
                if d not in groups_current[i]:
                    groups_current[i].append(d)

        groups += groups_current

    # check that intersection of all groups is empty
    assert not set.intersection(*[set(x) for x in groups])

    rename_dict = dict(sum([[(dim, 'new_'+group[0])
                             for dim in group]
                            for group in groups
                            if len(group) > 1  # don't rename if not useful
                            ], []))
    return A.rename_dims(rename_dict)
            

