#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for exploiting eoread objects
'''

import numpy as np
import xarray as xr
from numpy import cos, radians, sqrt

# backward compatibility:
from eoread.utils.save import to_netcdf # noqa
from eoread.utils.tools import datetime  # noqa
from eoread.utils.tools import (contains, getflag, haversine, locate,# noqa
                                merge, raiseflag, split, sub, sub_pt, sub_rect,
                                wrap)

from .utils.naming import naming


def init_Rtoa(ds: xr.Dataset):
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


def init_geometry(ds: xr.Dataset, 
                  scat_angle: bool =False):
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