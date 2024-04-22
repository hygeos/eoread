from eoread.reader.hdf4 import load_hdf4
from eoread.utils.tools import merge
from eoread.utils.config import load_config
from eoread.utils.naming import naming as n
from eoread.download.download_nextcloud import download_nextcloud
from pathlib import Path

import xarray as xr
import dask.array as da



bands_250 = [650,860]
bands_500 = [470,555,1240,1640,2130]
bands_vis = [410,440,485,530,550,668,670,680,685,750,870,900,935,940,1375]
bands_tir = [3750,3960,4050,4460,4510,1375,6710,7230,8550,9730,11000,12000,13230,13630,13930,14230]

band_index = { # Bands      - wavelength (um)   - resolution (m)    - group
    650:  1,   # Band 1        0.62 - 0.67	         250              250m
    860:  2,   # Band 2        0.84 - 0.87	         250              250m
    470:  3,   # Band 3        0.46 - 0.48	         500              500m
    555:  4,   # Band 4        0.54 - 0.56           500              500m
    1240: 5,   # Band 5   	   1.23 - 1.25	         500              500m
    1640: 6,   # Band 6   	   1.63 - 1.65	         500              500m
    2130: 7,   # Band 7   	   2.11 - 2.16	         500              500m
    410:  8,   # Band 8   	   0.40 - 0.42	         1000            Ref_1km
    440:  9,   # Band 9   	   0.44 - 0.45	         1000            Ref_1km
    485:  10,  # Band 10  	   0.48 - 0.49	         1000            Ref_1km
    530:  11,  # Band 11  	   0.52 - 0.53 	         1000            Ref_1km
    550:  12,  # Band 12  	   0.54 - 0.56	         1000            Ref_1km
    670:  13,  # Band 13  	   0.66 - 0.67	         1000            Ref_1km
    672:  13.5,# Band 13  	   0.66 - 0.67	         1000            Ref_1km
    680:  14,  # Band 14  	   0.67 - 0.68	         1000            Ref_1km
    682:  14.5,# Band 14  	   0.67 - 0.68	         1000            Ref_1km
    750:  15,  # Band 15  	   0.74 - 0.75 	         1000            Ref_1km
    870:  16,  # Band 16  	   0.86 - 0.88 	         1000            Ref_1km
    900:  17,  # Band 17  	   0.89 - 0.92	         1000            Ref_1km
    935:  18,  # Band 18  	   0.93 - 0.94	         1000            Ref_1km
    940:  19,  # Band 19  	   0.91 - 0.96	         1000            Ref_1km
    3750: 20,  # Band 20  	   3.66 - 3.84	         1000            Emi_1km
    3960: 21,  # Band 21  	   3.93 - 3.99	         1000            Emi_1km
    3962: 22,  # Band 22  	   3.93 - 3.99	         1000            Emi_1km
    4050: 23,  # Band 23  	   4.02 - 4.08	         1000            Emi_1km
    4460: 24,  # Band 24  	   4.43 - 4.50	         1000            Emi_1km
    4510: 25,  # Band 25       4.48 - 4.55	         1000            Emi_1km
    1375: 26,  # Band 26 	   1.36 - 1.39	         1000            Ref_1km
    6710: 27,  # Band 27 	   6.53 - 6.89	         1000            Emi_1km
    7230: 28,  # Band 28  	   7.17 - 7.47	         1000            Emi_1km
    8550: 29,  # Band 29  	   8.40 - 8.70	         1000            Emi_1km
    9730: 30,  # Band 30  	   9.58 - 9.88	         1000            Emi_1km
    11000:31,  # Band 31  	   10.78 - 10.28         1000            Emi_1km
    12000:32,  # Band 32  	   11.77 - 12.27         1000            Emi_1km
    13230:33,  # Band 33       13.18 - 13.48         1000            Emi_1km
    13630:34,  # Band 34       13.48 - 13.78         1000            Emi_1km
    13930:35,  # Band 35       13.78 - 14.08         1000            Emi_1km
    14230:36,  # Band 36       14.08 - 14.38         1000            Emi_1km
    }

# Planck's law constants 
h = 6.6260755e-34
c = 2.9979246e+8 # (meters per second)
k = 1.380658e-23 # (Joules per Kelvin)

# derived constants
K1 = 2.0 * h * c * c
K2 = h * c / k


def Level1_MODIS(filepath: Path | str,
                 radiometry: str ='reflectance',
                 chunks: int = 500,
                 split: bool = False):
    # Revize variables
    filepath = Path(filepath)
    raw = load_hdf4(filepath, trim_dims=True, chunks=chunks)
    coords = raw.coords
    reverse_band_index = {v:k for k,v in band_index.items()}
    keep_vars = ['Latitude', 'Longitude', 'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth', 'gflags']
    new_vars  = [n.lat, n.lon, n.vza, n.vaa, n.sza, n.saa, n.flags]
    drop_vars = [n for n in list(raw.variables.keys()) if n not in keep_vars]
    l1 = raw.drop_vars(drop_vars)
    l1 = l1.rename_vars(dict(zip(keep_vars, new_vars)))

    # Rescale angles data
    for varname in new_vars[2:-1]:
        l1[varname] = l1[varname].scale_factor * l1[varname]

    # Change radiometry of input data   
    l1 = transform_radiometry(raw, l1, radiometry, split)

    # Change dimensions name and update coordinates
    old_dims = ['2*nscans:MODIS_SWATH_Type_L1B', '1KM_geo_dim:MODIS_SWATH_Type_L1B', 
                '10*nscans:MODIS_SWATH_Type_L1B', 'Max_EV_frames:MODIS_SWATH_Type_L1B']
    new_dims = ['y_red','x_red',n.rows,n.columns]
    new_coords = {}
    if not split:
        new_dims = new_dims + [n.bands,n.bands_tir]
        old_dims = old_dims + ['bands_vis','bands_bt']
        new_coords[n.bands_tir] = [reverse_band_index[i] for i in coords['Band_1KM_Emissive'].values]
        new_coords[n.bands] = [reverse_band_index[i] for i in coords['Band_250M'].values] + \
                              [reverse_band_index[i] for i in coords['Band_500M'].values] + \
                              [reverse_band_index[i] for i in coords['Band_1KM_RefSB'].values]
    
    revize_dims = dict(zip(old_dims, new_dims))
    l1 = l1.rename_dims(revize_dims)
    l1 = l1.assign_coords(new_coords)

    # Summarize Attributes
    list_attr = [attr.split("=") for attr in l1.attrs['CoreMetadata.0'].split('\n') if len(attr) != 0]
    attributes = {}
    l1.attrs = {}
    parse_attrs(list_attr, attributes)
    l1.attrs[n.input_directory] = str(filepath.parent)
    l1.attrs[n.resolution]   = 1000
    l1.attrs[n.datetime]     = attributes['INVENTORYMETADATA']['ECSDATAGRANULE']['PRODUCTIONDATETIME'][1:-1]
    l1.attrs['night']        = str(attributes['INVENTORYMETADATA']['ECSDATAGRANULE']['DAYNIGHTFLAG'] != '"Day"')
    l1.attrs[n.product_name] = attributes['INVENTORYMETADATA']['ECSDATAGRANULE']['LOCALGRANULEID'][1:-1]
    l1.attrs[n.platform]     = attributes['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMSHORTNAME'][1:-1]
    l1.attrs[n.sensor]       = attributes['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER'][1:-1]
    l1.attrs[n.shortname]    = attributes['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['SHORTNAME'][1:-1]
    l1.attrs['version']      = int(attributes['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['VERSIONID'])

    return l1

def transform_radiometry(raw_data, level1, radiometry, split):
    assert radiometry in ['radiance','reflectance'], \
        f'Invalid radiometry value, get {radiometry}'
    
    toa = n.Ltoa if radiometry == 'radiance' else n.Rtoa
    bt  = n.Ltoa_tir if radiometry == 'radiance' else n.BT
    tag_scale  = f'{radiometry}_scales'
    tag_offset = f'{radiometry}_offsets'
    tag_unit   = f'{radiometry}_units'
    new_varname = [toa+'_250',toa+'_500',toa+'_1km',bt] if split else [toa,toa,toa,bt]
    old_varname = ['EV_250_Aggr1km_RefSB','EV_500_Aggr1km_RefSB','EV_1KM_RefSB','EV_1KM_Emissive']

    # Process Reflective bands
    cursor = 0
    size = raw_data['EV_1KM_Emissive'][0].shape
    upscale_sza = da.repeat(da.repeat(level1.sza.data,5,axis=0),5,axis=1)
    upscale_sza = upscale_sza[:size[0],:size[1]].rechunk(chunks=500)
    for var,new in zip(old_varname[:-1], new_varname[:-1]):
        ref_b   = raw_data[var]
        scales  = ref_b.attrs[tag_scale]
        offsets = ref_b.attrs[tag_offset]
        unit    = ref_b.attrs[tag_unit]
        for i,band in enumerate(ref_b):
            band.attrs = {}
            level1[new+f'_{cursor+1}'] = scales[i] * (band - offsets[i])
            level1[new+f'_{cursor+1}'].attrs['unit'] = unit
            if radiometry == 'reflectance':
                level1[new+f'_{cursor+1}'] /= da.cos(da.radians(upscale_sza))
            cursor += 1
    if not split:
        level1 = merge(level1, dim=f'bands_vis', pattern=r'(.+)_(\d+)')
        level1[new].attrs['unit'] = unit
    
    # Process Emissive bands
    emi_b   = raw_data['EV_1KM_Emissive']
    scales  = emi_b.attrs['radiance_scales']
    offsets = emi_b.attrs['radiance_offsets']
    unit    = emi_b.attrs['radiance_units']
    for i,band in enumerate(emi_b):
        band.attrs = {}
        level1[bt+f'_{i+1}'] = scales[i] * (band - offsets[i])
        level1[bt+f'_{i+1}'].attrs['unit'] = 'Kelvin'
        if radiometry == 'reflectance':
            level1[bt+f'_{i+1}'] = calibrate_bt(level1[bt+f'_{i+1}'],i)
    if not split:
        level1 = merge(level1, dim='bands_bt', pattern=r'(.+)_(\d+)')
        level1[bt].attrs['unit'] = 'Kelvin' if radiometry == 'reflectance' else unit
        level1[n.flags] = level1[bt].isel(bands_bt=0).isnull().astype(n.flags_dtype)
    
    level1 = level1.drop_indexes(list(level1.coords)) \
                   .reset_coords(drop=True)
    return level1

def calibrate_bt(array, band_index):
    """Calibration for the emissive channels."""

    # Planck's law constants 
    h = 6.6260755e-34
    c = 2.9979246e+8 # (meters per second)
    k = 1.380658e-23 # (Joules per Kelvin)

    # derived constants
    K1 = 2.0 * h * c * c
    K2 = h * c / k

    # Effective central wavenumber (inverse centimeters)
    cwn = da.array([
        2.641775E+3, 2.505277E+3, 2.518028E+3, 2.465428E+3,
        2.235815E+3, 2.200346E+3, 1.477967E+3, 1.362737E+3,
        1.173190E+3, 1.027715E+3, 9.080884E+2, 8.315399E+2,
        7.483394E+2, 7.308963E+2, 7.188681E+2, 7.045367E+2],
        dtype=float)

    # Temperature correction slope (no units)
    tcs = da.array([
        9.993411E-1, 9.998646E-1, 9.998584E-1, 9.998682E-1,
        9.998819E-1, 9.998845E-1, 9.994877E-1, 9.994918E-1,
        9.995495E-1, 9.997398E-1, 9.995608E-1, 9.997256E-1,
        9.999160E-1, 9.999167E-1, 9.999191E-1, 9.999281E-1],
        dtype=float)

    # Temperature correction intercept (Kelvin)
    tci = da.array([
        4.770532E-1, 9.262664E-2, 9.757996E-2, 8.929242E-2,
        7.310901E-2, 7.060415E-2, 2.204921E-1, 2.046087E-1,
        1.599191E-1, 8.253401E-2, 1.302699E-1, 7.181833E-2,
        1.972608E-2, 1.913568E-2, 1.817817E-2, 1.583042E-2],
        dtype=float)

    # Transfer wavenumber [cm^(-1)] to wavelength [m]
    cwvl = 1. / (cwn * 100)

    # Some versions of the modis files do not contain all the bands.
    cwvl = cwvl[band_index]
    tcs  = tcs[band_index]
    tci  = tci[band_index]
    array = K2 / (cwvl * da.log(K1 / (1e6 * array * cwvl ** 5) + 1))
    array = (array - tci) / tcs
    return array

def parse_attrs(stack, out_dic={}):
    current = [elem.strip() for elem in stack[0]]
    if current[0] == 'END_GROUP':
        return out_dic, stack
    elif current[0] == 'GROUP':
        out_dic[current[1]], new_stack = parse_attrs(stack[1:],{})
        return parse_attrs(new_stack[1:], out_dic)
    elif current[0] == 'OBJECT':
        for i in range(10):
            sub = [elem.strip() for elem in stack[i+1]]
            if sub[0] == 'VALUE':
                out_dic[current[1]] = sub[1]
            if 'END' in sub[0]:
                break
        return parse_attrs(stack[i+2:],out_dic)
    else:
        return parse_attrs(stack[1:], out_dic)
    

def get_sample():
    product_name = 'MYD021KM_example.hdf'
    return download_nextcloud(product_name, load_config()['dir_samples'], 'SampleData')