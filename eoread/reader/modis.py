from .hdf4 import load_hdf4



def Level1_MODIS(filepath,
                 chunks=500,
                 split=False):
    # Revize variables
    l1 = load_hdf4(filepath, trim_dims=True, chunks=chunks)
    keep_vars = ['Latitude', 'Longitude', 'EV_1KM_RefSB', 'EV_1KM_Emissive', 'EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB', 'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth', 'gflags']
    new_vars  = ['Latitude', 'Longitude', 'Rtoa', 'BT', 'Rtoa_250', 'Rtoa_500', 'vza', 'vaa', 'sza', 'saa', 'flags']
    drop_vars = [n for n in list(l1.variables.keys()) if n not in keep_vars]
    l1 = l1.drop_vars(drop_vars)
    l1 = l1.rename_vars(dict(zip(keep_vars, new_vars)))

    # Change dimensions name
    new_dims = ['x_red','y_red','bands','x','y','bands_tir','bands_250','bands_500']
    revize_dims = dict(zip(list(l1.dims), new_dims))
    l1 = l1.rename_dims(revize_dims)

    # Update coordinates
    coords = {'bands_tir':[3750,3960,4050,4460,4510,1375,6710,7230,8550,9730,11000,12000,13230,13630,13930,14230],
            'bands'    :[410,440,485,530,550,668,670,680,685,750,870,900,935,940,1375],
            'band_500' :[470,555,1240,1640,2130],
            'band_250' :[650,860]}
    l1 = l1.assign_coords(coords)

    # Summarize Attributes
    list_attr = [attr.split("=") for attr in l1.attrs['CoreMetadata.0'].split('\n') if len(attr) != 0]
    attributes = {}
    l1.attrs = {}
    parse_attrs(list_attr, attributes)
    l1.attrs['datetime'] = attributes['INVENTORYMETADATA']['ECSDATAGRANULE']['PRODUCTIONDATETIME'][1:-1]
    l1.attrs['night'] = attributes['INVENTORYMETADATA']['ECSDATAGRANULE']['DAYNIGHTFLAG'] != '"Day"'
    l1.attrs['granule_id'] = attributes['INVENTORYMETADATA']['ECSDATAGRANULE']['LOCALGRANULEID'][1:-1]
    l1.attrs['platform'] = attributes['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMSHORTNAME'][1:-1]
    l1.attrs['sensor'] = attributes['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER'][1:-1]
    l1.attrs['shortname'] = attributes['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['SHORTNAME'][1:-1]
    l1.attrs['version'] = int(attributes['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['VERSIONID'])


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