#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import xarray as xr
import dask.array as da
import os
import numpy as np
from xml.dom.minidom import parse, parseString
from datetime import datetime

from eoread.fileutils import mdir

from . import eo
from .common import Interpolator, AtIndex
from .naming import naming, flags
from .common import DataArray_from_array


olci_band_names = {
        '01': 400, '02': 412,
        '03': 443, '04': 490,
        '05': 510, '06': 560,
        '07': 620, '08': 665,
        '09': 674, '10': 681,
        '11': 709, '12': 754,
        '13': 760, '14': 764,
        '15': 767, '16': 779,
        '17': 865, '18': 885,
        '19': 900, '20': 940,
        '21': 1020,
    }

# central wavelength of the detector (for normalization)
# (detector 374 of camera 3)
central_wavelength_olci = {
    400: 400.664, 412: 412.076,
    443: 443.183, 490: 490.713,
    510: 510.639, 560: 560.579,
    620: 620.632, 665: 665.3719,
    674: 674.105, 681: 681.66,
    709: 709.1799, 754: 754.2236,
    760: 761.8164, 764: 764.9075,
    767: 767.9734, 779: 779.2685,
    865: 865.4625, 885: 884.3256,
    900: 899.3162, 940: 939.02,
    1020: 1015.9766, 1375: 1375.,
    1610: 1610., 2250: 2250.,
}


def get_sample(kind: str, dir_samples=None) -> Path:
    from eoread.download_eumdac import download_eumdac
    
    pname = {
        # France
        'level1_fr': 'S3B_OL_1_EFR____20220616T101508_20220616T101808_20220617T153119'
                     '_0179_067_122_2160_MAR_O_NT_002.SEN3',
        # Rio de la Plata
        'level2_fr': 'S3A_OL_2_WFR____20240115T131414_20240115T131714_20240115T145301'
                     '_0179_108_038_3600_MAR_O_NR_003.SEN3'
    }[kind]

    target = mdir(dir_samples or naming.dir_samples)/pname
    download_eumdac(target)
    return target


def Level1_OLCI(dirname,
                chunks=500,
                tie_param=False,
                init_spectral=True,
                init_reflectance=False,
                interp_angles='linear',
                ):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    '''
    ds = read_OLCI(dirname,
                   level='level1',
                   chunks=chunks,
                   tie_param=tie_param,
                   init_spectral=(init_spectral or init_reflectance),
                   interp_angles=interp_angles,
                   )

    if init_reflectance:
        eo.init_Rtoa(ds)

    return ds.unify_chunks()



def Level2_OLCI(dirname,
                chunks=500,
                tie_param=False,
                init_spectral=True,
                interp_angles='linear',
                ):
    '''
    Read an OLCI Level2 product as an xarray.Dataset
    '''
    return read_OLCI(dirname,
                     level='level2',
                     chunks=chunks,
                     tie_param=tie_param,
                     init_spectral=init_spectral,
                     interp_angles=interp_angles,
                     )


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

    footprint_lat, footprint_lon = zip(*footprint)

    return {'bandfilenames': bandfilenames,
            'footprint_lat': footprint_lat,
            'footprint_lon': footprint_lon,
            'textinfo': textinfo,
            }


def read_OLCI(dirname,
              chunks=None,
              level=None,
              tie_param=False,
              init_spectral=False,
              engine=None,
              interp_angles='linear',
              ):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances, the angles on the full grid, etc.
    
    interp_angles:
        'linear': linear interpolation
        'atan2': interpolate sin(x) and cos(x), then x = atan2(sin, cos)
        'legacy': for backward compatibility (nearest for azimuth angles, linear for zenith angles)
    '''
    ds = xr.Dataset()

    dirname = Path(dirname)
    if (dirname/dirname.name).exists():
        dirname = (dirname/dirname.name)

    # read manifest file for file names and footprint
    manifest = read_manifest(dirname)
    ds.attrs[naming.footprint_lat] = manifest['footprint_lat']
    ds.attrs[naming.footprint_lon] = manifest['footprint_lon']

    try:
        level_from_manifest = {
                'SENTINEL-3 OLCI Level 1 Earth Observation Full Resolution Product': 'level1',
                'SENTINEL-3 OLCI Level 1 Earth Observation Reduced Resolution Product': 'level1',
                'SENTINEL-3 OLCI Level 2 Water Product': 'level2',
                }[manifest['textinfo']]
    except KeyError:
        raise Exception('Invalid textinfo in manifest: "{}"'.format(manifest['textinfo']))
    assert (level is None) or (level == level_from_manifest), \
        f'expected {level} encountered {level_from_manifest}'

    # Read main product
    prod_list = []
    bands = []
    for idx, filename in manifest['bandfilenames']:
        if '_unc' in filename:
            continue
        fname = os.path.join(dirname, filename)
        prod_list.append(xr.open_dataset(fname, chunks=chunks, engine=engine)[os.path.basename(fname)[:-3]])
        bands.append(olci_band_names[idx])

    index_bands = xr.IndexVariable('bands', bands)
    if level == 'level1':
        param_name = naming.Ltoa
    else:
        param_name = naming.Rw
    ds[param_name] = xr.concat(prod_list, dim=index_bands)

    # Geo coordinates
    geo_coords_file = os.path.join(dirname, 'geo_coordinates.nc')
    geo = xr.open_dataset(geo_coords_file, chunks=chunks, engine=engine)
    for k in geo.variables:
        ds[k] = geo[k].astype('float32')
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
    assert dims3 == ds[param_name].dims

    # tie geometry interpolation
    tie_geom_file = os.path.join(dirname, 'tie_geometries.nc')
    tie_ds = xr.open_dataset(tie_geom_file, chunks=-1, engine=engine)
    tie_ds = tie_ds.assign_coords(
                tie_columns=np.arange(tie_ds.dims['tie_columns'])*ds.ac_subsampling_factor,
                tie_rows=np.arange(tie_ds.dims['tie_rows'])*ds.al_subsampling_factor,
                )
    assert tie_ds.tie_columns[0] == ds.columns[0]
    assert tie_ds.tie_columns[-1] == ds.columns[-1]
    assert tie_ds.tie_rows[0] == ds.rows[0]
    assert tie_ds.tie_rows[-1] == ds.rows[-1]

    if interp_angles == 'linear':
        interp_aa = 'linear'
        interp_za = 'linear'
    elif interp_angles == 'atan2':
        interp_aa = 'atan2'
        interp_za = 'atan2'
    elif interp_angles == 'legacy':
        interp_aa = 'nearest'
        interp_za = 'linear'
    else:
        raise ValueError(f'Invalid interp_angles "{interp_angles}"')
    
    for (ds_full, ds_tie, method) in [
                ('sza', 'SZA', interp_za),
                ('saa', 'SAA', interp_aa),
                ('vza', 'OZA', interp_za),
                ('vaa', 'OAA', interp_aa),
            ]:
        if method == 'atan2':
            _cos = DataArray_from_array(
                Interpolator(shape2, np.cos(np.radians(tie_ds[ds_tie].astype('float32'))), 'linear'),
                dims2,
                chunks,
            )
            _sin = DataArray_from_array(
                Interpolator(shape2, np.sin(np.radians(tie_ds[ds_tie].astype('float32'))), 'linear'),
                dims2,
                chunks,
            )
            ds[ds_full] = np.degrees(np.arctan2(_sin, _cos))
        else:
            ds[ds_full] = DataArray_from_array(
                Interpolator(shape2, tie_ds[ds_tie].astype('float32'), method),
                dims2,
                chunks,
            )
        ds[ds_full].attrs = tie_ds[ds_tie].attrs
        if tie_param:
            ds[ds_full+'_tie'] = tie_ds[ds_tie]

    # tie meteo interpolation
    tie_meteo_file = os.path.join(dirname, 'tie_meteo.nc')
    tie = xr.open_dataset(tie_meteo_file, chunks=-1, engine=engine)
    tie = tie.assign_coords(
                tie_columns = np.arange(tie.dims['tie_columns'])*ds.ac_subsampling_factor,
                tie_rows = np.arange(tie.dims['tie_rows'])*ds.al_subsampling_factor,
                )
    assert tie.tie_columns[0] == ds.columns[0]
    assert tie.tie_columns[-1] == ds.columns[-1]
    assert tie.tie_rows[0] == ds.rows[0]
    assert tie.tie_rows[-1] == ds.rows[-1]
    
    wind0 = DataArray_from_array(
        Interpolator(
            shape2,
            tie.horizontal_wind.isel(wind_vectors=0)
        ),
        dims2,
        chunks,
    )
    wind1 = DataArray_from_array(
        Interpolator(
            shape2,
            tie.horizontal_wind.isel(wind_vectors=1)
        ),
        dims2,
        chunks,
    )
    ds[naming.horizontal_wind] = np.sqrt(wind0**2 + wind1**2)
    ds[naming.horizontal_wind].attrs = tie[naming.horizontal_wind].attrs
    variables = [
        'humidity',
        naming.sea_level_pressure,
        naming.total_columnar_water_vapour,
        naming.total_ozone]
    for var in variables:
        ds[var] = DataArray_from_array(
            Interpolator(shape2, tie[var]),
            dims2,
            chunks,
        )
        ds[var].attrs = tie[var].attrs
        if tie_param:
            ds[var+'_tie'] = tie[var]

    # check subsampling factors
    assert (ds.dims['columns']-1) == ds.ac_subsampling_factor*(tie_ds.dims['tie_columns']-1)
    assert (ds.dims['rows']-1) == ds.al_subsampling_factor*(tie_ds.dims['tie_rows']-1)

    # instrument data
    instrument_data_file = os.path.join(dirname, 'instrument_data.nc')
    instrument_data = xr.open_dataset(instrument_data_file,
                                      chunks=chunks,
                                      mask_and_scale=False,
                                      # this variable has duplicate dimensions, drop it
                                      drop_variables='relative_spectral_covariance',
                                      engine=engine)
    if level == 'level2':
        instrument_data = instrument_data.rename({'bands': 'bands_full'})
        bands_full = list(olci_band_names.values())
        assert bands_full == sorted(bands_full)
        instrument_data = instrument_data.assign_coords(bands_full=bands_full)
    for x in instrument_data.variables:
        ds[x] = instrument_data[x]

    if level == 'level1':
        # quality flags
        qf_file = os.path.join(dirname, 'qualityFlags.nc')
        qf = xr.open_dataset(qf_file, chunks=chunks, engine=engine)
        ds['quality_flags'] = qf.quality_flags
    else:
        # chl_nn
        fname = os.path.join(dirname, 'chl_nn.nc')
        qf = xr.open_dataset(fname, chunks=chunks, engine=engine)
        ds['chl_nn'] = qf.CHL_NN

        # chl_oc4me
        fname = os.path.join(dirname, 'chl_oc4me.nc')
        qf = xr.open_dataset(fname, chunks=chunks, engine=engine)
        ds['chl_oc4me'] = qf.CHL_OC4ME

        # quality flags
        fname = os.path.join(dirname, 'wqsf.nc')
        qf = xr.open_dataset(fname, chunks=chunks, engine=engine)
        ds['wqsf'] = qf.WQSF

        # aerosol properties
        fname = os.path.join(dirname, 'w_aer.nc')
        qf = xr.open_dataset(fname, chunks=chunks, engine=engine)
        ds['A865'] = qf.A865
        ds['T865'] = qf.T865

    # flags
    if level == 'level1':
        ds[naming.flags] = xr.zeros_like(
            ds.vza,
            dtype=naming.flags_dtype)
        qf = eo.getflags(ds.quality_flags)
        eo.raiseflag(ds[naming.flags],
                    'LAND',
                    flags['LAND'],
                    ds.quality_flags & qf['land'])
        eo.raiseflag(ds[naming.flags],
                    'L1_INVALID',
                    flags['L1_INVALID'],
                    ds.quality_flags & qf['invalid'])

    # attributes
    dstart = datetime.strptime(ds.start_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    dstop = datetime.strptime(ds.stop_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    ds.attrs[naming.datetime] = (dstart + (dstop - dstart)/2.).isoformat()
    ds.attrs[naming.platform] = 'Sentinel-3'   # FIXME: A or B
    ds.attrs[naming.sensor] = 'OLCI'
    ds.attrs[naming.input_directory] = os.path.dirname(dirname)

    ds = ds.chunk(dict(detectors=-1))   # FIXME: do this upstream

    if init_spectral:
        olci_init_spectral(ds, chunks)

    return ds

def olci_init_spectral(ds, chunks):
    '''
    Broadcast all spectral (detector-wise) dataset to the whole image

    Adds the resulting datasets to `ds`: wav, F0 (in place)
    '''
    # wavelength
    ds[naming.wav] = xr.apply_ufunc(
        lambda l0, di: l0[:,0,0,di],
        ds.lambda0,  # (bands x detectors)
        ds.detector_index,   # (rows x columns)
        dask='parallelized',
        input_core_dims=[['detectors'], []],
        output_dtypes=[ds.lambda0.dtype],
    )
    ds[naming.wav].attrs.update(ds.lambda0.attrs)

    # solar flux
    ds[naming.F0] = xr.apply_ufunc(
        lambda sf, di: sf[:,0,0,di],
        ds.solar_flux,  # (bands x detectors)
        ds.detector_index,   # (rows x columns)
        dask='parallelized',
        input_core_dims=[['detectors'], []],
        output_dtypes=[ds.solar_flux.dtype],
    )
    ds[naming.F0].attrs.update(ds.solar_flux.attrs)

    # central (nominal) wavelength
    ds[naming.cwav] = xr.DataArray(
        np.array([central_wavelength_olci[b] for b in ds.bands.data],
                 dtype='float32'),
        dims=('bands',))


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
    L2_FLAGS = eo.getflags(wqsf)
    for flag in flags:
        bval += int(L2_FLAGS[flag])

    return wqsf & bval == 0
