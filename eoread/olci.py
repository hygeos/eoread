#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import xarray as xr
import dask.array as da
import os
import numpy as np
from eoread.common import Interpolator, AtIndex
from xml.dom.minidom import parse, parseString


olci_band_names = {
        '01': 400 , '02': 412,
        '03': 443 , '04': 490,
        '05': 510 , '06': 560,
        '07': 620 , '08': 665,
        '09': 674 , '10': 681,
        '11': 709 , '12': 754,
        '13': 760 , '14': 764,
        '15': 767 , '16': 779,
        '17': 865 , '18': 885,
        '19': 900 , '20': 940,
        '21': 1020,
    }


def Level1_OLCI(dirname, chunks={'columns': 400, 'rows': 300}, tie_param=False, init_spectral=True):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    '''
    return read_OLCI(dirname, level='level1', chunks=chunks, tie_param=tie_param, init_spectral=init_spectral)


def Level2_OLCI(dirname, chunks={'columns': 400, 'rows': 300}, tie_param=False, init_spectral=True):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    '''
    return read_OLCI(dirname, level='level2', chunks=chunks, tie_param=tie_param, init_spectral=init_spectral)


def read_manifest(dirname):
    # parse file
    filename = os.path.join(dirname, 'xfdumanifest.xml')
    bandfilenames = []  # mapping index -> filename
    with open(filename) as pf:
        manif = pf.read()
    dom = parseString(manif)
    
    # read product type
    textinfo = dom.getElementsByTagName('xfdu:contentUnit')[0].attributes['textInfo'].value
    
    # read bands and related files
    for n in dom.getElementsByTagName('dataObject'):
        inode = n.attributes['ID'].value[:-4]
        href = n.getElementsByTagName('fileLocation')[0].attributes['href'].value
        if inode.startswith('Oa'):
            bandfilenames.append((inode[2:4], href))

    # read footprint
    n = dom.getElementsByTagName('sentinel-safe:footPrint')[0]
    footprint = n.getElementsByTagName('gml:posList')[0].lastChild.data
    idata = iter(footprint.split())
    footprint = [(float(v), float(idata.__next__())) for v in idata]

    return {'bandfilenames': bandfilenames,
            'footprint': footprint,
            'textinfo': textinfo,
            }

def read_OLCI(dirname, level=None, chunks={'columns': 400, 'rows': 300}, tie_param=False, init_spectral=True):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances, the angles on the full grid, etc.

    '''
    ds = xr.Dataset()

    # read manifest file for file names and footprint
    manifest = read_manifest(dirname)
    ds.attrs['Footprint'] = manifest['footprint']

    level_from_manifest = {
            'SENTINEL-3 OLCI Level 1 Earth Observation Full Resolution Product': 'level1',
            'SENTINEL-3 OLCI Level 2 Water Product': 'level2',
            }[manifest['textinfo']]
    assert (level is None) or (level == level_from_manifest), ('expected', level, 'encountered', level_from_manifest)

    # Read TOA radiance
    prod_list = []
    bands = []
    for idx, filename in manifest['bandfilenames']:
        fname = os.path.join(dirname, filename)
        prod_list.append(xr.open_dataset(fname, chunks=chunks)[os.path.basename(fname)[:-3]])
        bands.append(olci_band_names[idx])

    index_bands = xr.IndexVariable('bands', bands)
    if level == 'level1':
        param_name = 'Ltoa'
    else:
        param_name = 'Rw'
    ds[param_name] = xr.concat(prod_list, dim=index_bands)

    # Geo coordinates
    geo_coords_file = os.path.join(dirname, 'geo_coordinates.nc')
    geo = xr.open_dataset(geo_coords_file, chunks=chunks)
    for k in geo.variables:
        ds[k] = geo[k]
    ds.attrs.update(geo.attrs)

    # dimensions
    dims2 = ('rows', 'columns')
    dims3 = ('bands', 'rows', 'columns')

    assert dims2 == ds.latitude.dims
    shape2 = ds.latitude.shape
    chunksize2 = ds.latitude.data.chunksize
    assert dims3 == ds[param_name].dims

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
        if tie_param:
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
    if level == 'level2':
        instrument_data = instrument_data.rename({'bands': 'bands_full'})
    for x in instrument_data.variables:
        ds[x] = instrument_data[x]

    if init_spectral:
        olci_init_spectral(ds)

    # quality flags
    if level == 'level1':
        qf_file = os.path.join(dirname, 'qualityFlags.nc')
        qf = xr.open_dataset(qf_file, chunks=chunks)
        ds['quality_flags'] = qf.quality_flags
    else:
        # quality flags
        qf_file = os.path.join(dirname, 'wqsf.nc')
        qf = xr.open_dataset(qf_file, chunks=chunks)
        ds['wqsf'] = qf.WQSF

        # aerosol properties
        qf_file = os.path.join(dirname, 'w_aer.nc')
        qf = xr.open_dataset(qf_file, chunks=chunks)
        ds['A865'] = qf.A865
        ds['T865'] = qf.T865

    # TODO: read date

    return ds

def olci_init_spectral(ds):
    
    # dimensions to be indexed by this object
    dims = sum([[x] if not x == 'detectors' else list(ds.detector_index.dims) for x in ds.lambda0.dims], [])
    # ... and their chunksize
    if len(list(ds.lambda0.chunks))==0:
        # if DataSet not chunked
        chunksize = {}
    else:
        chunksize = sum([[ds.lambda0.data.chunksize[i]] if not x == 'detectors' else list(ds.detector_index.data.chunksize) for i, x in enumerate(ds.lambda0.dims)], [])

    # wavelength
    ds['wav'] = (dims, da.from_array(AtIndex(ds.lambda0,
                                              ds.detector_index,
                                              'detectors'),
                                      chunks=chunksize))
    ds['wav'].attrs.update(ds.lambda0.attrs)

    # solar flux
    ds['F0'] = (dims, da.from_array(AtIndex(ds.solar_flux,
                                              ds.detector_index,
                                              'detectors'),
                                      chunks=chunksize))
    ds['F0'].attrs.update(ds.solar_flux.attrs)