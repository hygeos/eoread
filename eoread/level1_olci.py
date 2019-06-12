#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import xarray as xr
import dask.array as da
import os
import numpy as np
from eoread.common import Interpolator, AtIndex
from xml.dom.minidom import parse, parseString


olci_band_names = [
    (400, 'Oa01'), (412, 'Oa02'),
    (443, 'Oa03'), (490, 'Oa04'),
    (510, 'Oa05'), (560, 'Oa06'),
    (620, 'Oa07'), (665, 'Oa08'),
    (674, 'Oa09'), (681, 'Oa10'),
    (709, 'Oa11'), (754, 'Oa12'),
    (760, 'Oa13'), (764, 'Oa14'),
    (767, 'Oa15'), (779, 'Oa16'),
    (865, 'Oa17'), (885, 'Oa18'),
    (900, 'Oa19'), (940, 'Oa20'),
    (1020, 'Oa21'),
    ]


def read_manifest(dirname):
    filename = '{}/xfdumanifest.xml'.format(dirname)
    bandfilenames = {}
    with open(filename) as pf:
        manif = pf.read()
        dom = parseString(manif)
        for n in dom.getElementsByTagName('dataObject'):
            inode = n.attributes['ID'].value[:-4]
            href = n.getElementsByTagName('fileLocation')[0].attributes['href'].value
            if '_radiance' in inode:
                bandfilenames[inode[:-9]] = href

        n = dom.getElementsByTagName('sentinel-safe:footPrint')[0]
        footprint = n.getElementsByTagName('gml:posList')[0].lastChild.data
        footprint = [float(v) for v in footprint.split()]

    return bandfilenames, footprint

def Level1_OLCI(dirname, chunks={'columns': 400, 'rows': 300}):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances, the angles on the full grid, etc.
    '''
    ds = xr.Dataset()

    # read manifest file for file names and footprint
    bandfilenames, footprint = read_manifest(dirname)

    # Read TOA radiance
    Ltoa_list = []
    for _, bname in olci_band_names:
        fname = os.path.join(dirname, bandfilenames[bname])
        Ltoa_list.append(xr.open_dataset(fname, chunks=chunks)[bname+'_radiance'])

    index_bands = xr.IndexVariable('bands', [k for k, _ in olci_band_names])
    ds['Ltoa'] = xr.concat(Ltoa_list, dim=index_bands)

    # Geo coordinates
    geo_coords_file = os.path.join(dirname, 'geo_coordinates.nc')
    geo = xr.open_dataset(geo_coords_file, chunks=chunks)
    for k in geo.variables:
        ds[k] = geo[k]
    ds.attrs.update(geo.attrs)

    # dimensions
    dims2 = ds.latitude.dims
    shape2 = ds.latitude.shape
    chunksize2 = ds.latitude.data.chunksize
    dims3 = ds.Ltoa.dims
    chunksize3 = ds.Ltoa.data.chunksize

    # tiepoint interpolation
    tie_geom_file = os.path.join(dirname, 'tie_geometries.nc')
    tie = xr.open_dataset(tie_geom_file, chunks={})
    tie = tie.assign_coords(
                tie_columns = np.arange(tie.dims['tie_columns'])*ds.ac_subsampling_factor,
                tie_rows = np.arange(tie.dims['tie_rows'])*ds.al_subsampling_factor,
                )
    assert tie.tie_columns[0] == ds.columns[0]
    assert tie.tie_columns[-1] == ds.columns[-1]
    assert tie.tie_rows[0] == ds.rows[0]
    assert tie.tie_rows[-1] == ds.rows[-1]
    for (ds_full, ds_tie, method) in [
                ('sza', 'SZA', 'linear'),
                ('saa', 'SAA', 'nearest'),
                ('vza', 'OZA', 'linear'),
                ('vaa', 'OAA', 'nearest'),
            ]:
        ds[ds_full] = (dims2, da.from_array(Interpolator(shape2, tie[ds_tie]),
                                            chunks=chunksize2))
        ds[ds_full].attrs = tie[ds_tie].attrs
        ds[ds_full+'_tie'] = tie[ds_tie]

    # check subsampling factors
    assert (ds.dims['columns']-1) == ds.ac_subsampling_factor*(tie.dims['tie_columns']-1)
    assert (ds.dims['rows']-1) == ds.al_subsampling_factor*(tie.dims['tie_rows']-1)

    # check lat/lon from tie
    if False:
        tie_geo_coords_file = os.path.join(dirname, 'tie_geo_coordinates.nc')
        tie_geo = xr.open_dataset(tie_geo_coords_file, chunks={})
        for (ds_full, ds_tie, method) in [
                    ('lat_from_tie', 'latitude', 'linear'),
                    ('lon_from_tie', 'longitude', 'nearest'),
                ]:
            ds[ds_full] = (dims2, da.from_array(
                tie_geo[ds_tie].interp(tie_rows=ds.rows/ds.al_subsampling_factor,
                                tie_columns=ds.columns/ds.ac_subsampling_factor,
                                method=method),
                chunks=chunksize2))

    # instrument data
    instrument_data_file = os.path.join(dirname, 'instrument_data.nc')
    instrument_data = xr.open_dataset(instrument_data_file, chunks=chunks, mask_and_scale=False)
    for x in instrument_data.variables:
        ds[x] = instrument_data[x]

    # wavelength
    ds['wav'] = (dims3, da.from_array(AtIndex(ds.lambda0,
                                              ds.detector_index,
                                              'detectors'),
                                      chunks=chunksize3))
    ds['wav'].attrs.update(ds.lambda0.attrs)

    # solar flux
    ds['F0'] = (dims3, da.from_array(AtIndex(ds.solar_flux,
                                              ds.detector_index,
                                              'detectors'),
                                      chunks=chunksize3))
    ds['F0'].attrs.update(ds.solar_flux.attrs)

    # quality flags
    qf_file = os.path.join(dirname, 'qualityFlags.nc')
    qf = xr.open_dataset(qf_file, chunks=chunks)
    ds['quality_flags'] = qf.quality_flags

    # TODO: read date

    return ds

