#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import xarray as xr
import dask.array as da
import os
import numpy as np
from eoread.common import Interpolator, AtIndex
from xml.dom.minidom import parse, parseString
from eoread.naming import Naming
from datetime import datetime


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


def Level1_OLCI(dirname, chunks={'columns': 400, 'rows': 300},
                tie_param=False, init_spectral=True, naming=Naming()):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    '''
    return read_OLCI(dirname, level='level1', chunks=chunks,
                     tie_param=tie_param, init_spectral=init_spectral, naming=naming)


def Level2_OLCI(dirname, chunks={'columns': 400, 'rows': 300},
                tie_param=False, init_spectral=True, naming=Naming()):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    '''
    return read_OLCI(dirname, level='level2', chunks=chunks,
                     tie_param=tie_param, init_spectral=init_spectral, naming=naming)


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

def read_OLCI(dirname, level=None, chunks={'columns': 400, 'rows': 300},
              tie_param=False, init_spectral=True, naming=Naming()):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances, the angles on the full grid, etc.

    '''
    ds = xr.Dataset()

    # read manifest file for file names and footprint
    manifest = read_manifest(dirname)
    ds.attrs['Footprint'] = manifest['footprint']

    try:
        level_from_manifest = {
                'SENTINEL-3 OLCI Level 1 Earth Observation Full Resolution Product': 'level1',
                'SENTINEL-3 OLCI Level 1 Earth Observation Reduced Resolution Product': 'level1',
                'SENTINEL-3 OLCI Level 2 Water Product': 'level2',
                }[manifest['textinfo']]
    except KeyError:
        raise Exception('Invalid textinfo in manifest: "{}"'.format(manifest['textinfo']))
    assert (level is None) or (level == level_from_manifest), ('expected', level, 'encountered', level_from_manifest)

    # Read main product
    prod_list = []
    bands = []
    for idx, filename in manifest['bandfilenames']:
        fname = os.path.join(dirname, filename)
        prod_list.append(xr.open_dataset(fname, chunks=chunks)[os.path.basename(fname)[:-3]])
        bands.append(olci_band_names[idx])

    index_bands = xr.IndexVariable('bands', bands)
    if level == 'level1':
        param_name = naming.Ltoa
    else:
        param_name = naming.Rw
    ds[param_name] = xr.concat(prod_list, dim=index_bands).chunk({naming.bands: -1})

    # Geo coordinates
    geo_coords_file = os.path.join(dirname, 'geo_coordinates.nc')
    geo = xr.open_dataset(geo_coords_file, chunks=chunks)
    for k in geo.variables:
        ds[k] = geo[k]
    ds.attrs.update(geo.attrs)

    # dimensions
    dims2 = naming.dim2
    dims3 = naming.dim3
    if level == 'level1':
        dims3_full = ('bands', 'rows', 'columns')
    else:
        dims3_full = ('bands_full', 'rows', 'columns')
    assert dims2 == ds.latitude.dims
    shape2 = ds.latitude.shape
    chunksize2 = ds.latitude.data.chunksize
    assert dims3 == ds[param_name].dims

    # tie geometry interpolation
    tie_geom_file = os.path.join(dirname, 'tie_geometries.nc')
    tie_ds = xr.open_dataset(tie_geom_file, chunks={})
    tie_ds = tie_ds.assign_coords(
                tie_columns = np.arange(tie_ds.dims['tie_columns'])*ds.ac_subsampling_factor,
                tie_rows = np.arange(tie_ds.dims['tie_rows'])*ds.al_subsampling_factor,
                )
    assert tie_ds.tie_columns[0] == ds.columns[0]
    assert tie_ds.tie_columns[-1] == ds.columns[-1]
    assert tie_ds.tie_rows[0] == ds.rows[0]
    assert tie_ds.tie_rows[-1] == ds.rows[-1]
    for (ds_full, ds_tie, method) in [
                ('sza', 'SZA', 'linear'),
                ('saa', 'SAA', 'nearest'),
                ('vza', 'OZA', 'linear'),
                ('vaa', 'OAA', 'nearest'),
            ]:
        ds[ds_full] = (dims2, da.from_array(Interpolator(shape2, tie_ds[ds_tie]),
                                            chunks=chunksize2))
        ds[ds_full].attrs = tie_ds[ds_tie].attrs
        if tie_param:
            ds[ds_full+'_tie'] = tie_ds[ds_tie]

    # tie meteo interpolation
    tie_meteo_file = os.path.join(dirname, 'tie_meteo.nc')
    tie = xr.open_dataset(tie_meteo_file, chunks={})
    tie = tie.assign_coords(
                tie_columns = np.arange(tie.dims['tie_columns'])*ds.ac_subsampling_factor,
                tie_rows = np.arange(tie.dims['tie_rows'])*ds.al_subsampling_factor,
                )
    assert tie.tie_columns[0] == ds.columns[0]
    assert tie.tie_columns[-1] == ds.columns[-1]
    assert tie.tie_rows[0] == ds.rows[0]
    assert tie.tie_rows[-1] == ds.rows[-1]
    
    ds['horizontal_wind'] = (dims2, da.from_array(Interpolator(shape2, np.sqrt(pow(tie.horizontal_wind.isel(wind_vectors=0),2)+pow(tie.horizontal_wind.isel(wind_vectors=1), 2))) ,chunks=chunksize2))
    ds['horizontal_wind'].attrs = tie['horizontal_wind'].attrs
    variables = ['humidity', 'sea_level_pressure', 'total_columnar_water_vapour', 'total_ozone']
    for var in variables:
        ds[var] = (dims2, da.from_array(Interpolator(shape2, tie[var]),
                                            chunks=chunksize2))
        ds[var].attrs = tie[var].attrs
        if tie_param:
            ds[var+'_tie'] = tie[var]

    # check subsampling factors
    assert (ds.dims['columns']-1) == ds.ac_subsampling_factor*(tie_ds.dims['tie_columns']-1)
    assert (ds.dims['rows']-1) == ds.al_subsampling_factor*(tie_ds.dims['tie_rows']-1)

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

    if level == 'level1':
        # quality flags
        qf_file = os.path.join(dirname, 'qualityFlags.nc')
        qf = xr.open_dataset(qf_file, chunks=chunks)
        ds['quality_flags'] = qf.quality_flags
    else:
        # chl_nn
        fname = os.path.join(dirname, 'chl_nn.nc')
        qf = xr.open_dataset(fname, chunks=chunks)
        ds['chl_nn'] = qf.CHL_NN

        # chl_oc4me
        fname = os.path.join(dirname, 'chl_oc4me.nc')
        qf = xr.open_dataset(fname, chunks=chunks)
        ds['chl_oc4me'] = qf.CHL_OC4ME

        # quality flags
        fname = os.path.join(dirname, 'wqsf.nc')
        qf = xr.open_dataset(fname, chunks=chunks)
        ds['wqsf'] = qf.WQSF

        # aerosol properties
        fname = os.path.join(dirname, 'w_aer.nc')
        qf = xr.open_dataset(fname, chunks=chunks)
        ds['A865'] = qf.A865
        ds['T865'] = qf.T865

    # attributes
    ds.attrs['datetime_start'] = datetime.strptime(ds.start_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    ds.attrs['datetime_stop'] = datetime.strptime(ds.stop_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    ds.attrs[naming.datetime] = ds.datetime_start + (ds.datetime_stop - ds.datetime_start)/2.
    ds.attrs[naming.platform] = 'Sentinel-3'   # FIXME: A or B
    ds.attrs[naming.sensor] = 'OLCI'

    return ds

def olci_init_spectral(ds):
    
    # dimensions to be indexed by this object
    dims = sum([[x] if not x == 'detectors' else list(ds.detector_index.dims) for x in ds.lambda0.dims], [])
    # ... and their chunksize
    if not ds.lambda0.chunks:
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

def get_l2_flags(wqsf):
    L2_FLAGS = dict(zip(
            wqsf.attrs['flag_meanings'].split(' '),
            [int(x) for x in wqsf.attrs['flag_masks']]))
    return L2_FLAGS

def decompose_flags(value, flags):
    '''
    return list of flag meanings for a given binary value
    flags: dictionary of meaning: value
    '''
    return [m for (m, v) in flags.items() if (v & value != 0)]


def get_valid_l2_pixels(wqsf, flags=[
        'INVALID', 'LAND', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN',
        'SNOW_ICE', 'SUSPECT', 'HISOLZEN', 'SATURATED', 'HIGHGLINT', 'WHITECAPS',
        'AC_FAIL', 'OC4ME_FAIL', 'ANNOT_TAU06', 'RWNEG_O2', 'RWNEG_O3', 'RWNEG_O4',
        'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8', 'ANNOT_ABSO_D',
        'ANNOT_DROUT', 'ANNOT_MIXR1']):
    '''
    Get valid standard level2 pixels with a given flag set
    '''
    bval = 0
    L2_FLAGS = get_l2_flags(wqsf)
    for flag in flags:
        bval += L2_FLAGS[flag]

    return wqsf & bval == 0



