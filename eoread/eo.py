#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Various utility functions for exploiting eoread objects
"""

import xarray as xr
import dask.array as da
from numpy import cos, radians, sqrt

# backward compatibility:
from core.files import to_netcdf
from core.tools import datetime
from core.tools import (contains, getflag, haversine, locate,
                        merge, raiseflag, split, sub, sub_pt, sub_rect,
                        wrap, getflags)
from core.geo import n

from warnings import warn
warn('The module `eo` will be deprecated from eoread package. Every remaining function will be moved to eotools.', DeprecationWarning)


def init_Rtoa(ds: xr.Dataset):
    """
    Initialize TOA reflectances from radiance (in place).

    Implies init_geometry.
    """
    init_geometry(ds)

    # TOA reflectance
    if str(n.rtoa) not in ds:
        ds = ds.assign({str(n.rtoa): ((str(n.bands),str(n.rows),str(n.columns)), 
                       (da.pi*ds[str(n.ltoa)]/(ds.mus*ds[str(n.F0)])).data)})
        ds[str(n.rtoa)].attrs.update(unit=None)

    return ds

def scattering_angle(mu_s, mu_v, phi):
    """
    Scattering angle in degrees.

    Parameters
    ----------
    mu_s : float or array_like
        Cosine of the sun zenith angle.
    mu_v : float or array_like
        Cosine of the view zenith angle.
    phi : float or array_like
        Relative azimuth angle in degrees.
    """
    sa = -mu_s*mu_v - sqrt((1.-mu_s*mu_s)*(1.-mu_v*mu_v)) * cos(radians(phi))
    return da.arccos(sa)*180./da.pi


def init_geometry(ds: xr.Dataset,
                  scat_angle: bool = False):
    """
    Initialize geometric variables (in place).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to modify in place.
    scat_angle : bool, optional
        If True, also compute the scattering angle. Default is False.
    """

    # mus and muv
    if str(n.mus) not in ds:
        ds[str(n.mus)] = da.cos(da.radians(ds.sza))
        ds[str(n.mus)].attrs['description'] = n.mus.desc
    if str(n.muv) not in ds:
        ds[str(n.muv)] = da.cos(da.radians(ds.vza))
        ds[str(n.muv)].attrs['description'] = n.muv.desc

    # relative azimuth angle
    if str(n.raa) not in ds:
        raa = ds[str(n.saa)] - ds[str(n.vaa)]
        raa = raa % 360
        ds[str(n.raa)] = raa.where(raa < 180, 360-raa)
        ds.raa.attrs['description'] = n.raa.desc
        ds.raa.attrs['unit'] = n.raa.unit

    # scattering angle
    if scat_angle:
        ds['scat_angle'] = scattering_angle(ds.mus, ds.muv, ds.raa)
        ds['scat_angle'].attrs['description'] = 'scattering angle'

    return ds


def show_footprint(ds: xr.Dataset, 
                   zoom: int = 4):
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