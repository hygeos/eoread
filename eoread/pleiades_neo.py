from pathlib import Path

import xarray as xr
from core.env import getdir
from core.network.download import download_url
from core.uncompress import uncompress_decorator
import rioxarray
from core.interpolate import interp, Linear
from core.geo import n
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from xmltodict import parse
from jrc_rayleigh_processor.utils import iter_through_subset

def parse_dimap_data(xml_file_path):
    """
    Parses a DIMAP XML file (Pleiades Neo) and extracts specific dataset, 
    raster, radiometric, and geometric information into a dictionary.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    output_data = {}

    # 1. Dataset Extent
    vertices = []
    dataset_extent_node = root.find(".//Dataset_Content/Dataset_Extent")
    if dataset_extent_node is not None:
        for vertex in dataset_extent_node.findall("Vertex"):
            v_data = {
                "LON": vertex.find("LON").text,
                "LAT": vertex.find("LAT").text,
                "COL": vertex.find("COL").text,
                "ROW": vertex.find("ROW").text
            }
            vertices.append(v_data)
    output_data["Dataset_Extent"] = vertices

    # 2. Identification & CRS
    ident_node = root.find(".//Product_Information/Delivery_Identification")
    output_data["Identification"] = {child.tag: child.text for child in ident_node}
    if "Order_Identification" in output_data["Identification"]:
        del output_data["Identification"]["Order_Identification"]
    
    source_node = root.find(".//Dataset_Sources/Source_Identification/Strip_Source")
    output_data["Source_ID"] = {child.tag: child.text for child in source_node}

    crs_node = root.find(".//Coordinate_Reference_System/Geodetic_CRS")
    output_data["Geodetic_CRS"] = {child.tag: child.text for child in crs_node if child.text} if crs_node is not None else {}

    # 3. Raster Data & File Mapping
    raster_data = {}
    dims_node = root.find(".//Raster_Data/Raster_Dimensions")
    raster_data["Raster_Dimensions"] = {child.tag: child.text for child in dims_node}

    special_values = []
    for sv in root.findall(".//Special_Value"):
        special_values.append({child.tag: child.text for child in sv})
    raster_data["Special_Values"] = special_values

    file_band_map = {}
    data_access = root.find(".//Raster_Data/Data_Access")
    
    for df_container in data_access.findall("Data_Files"):
        path_node = df_container.find(".//DATA_FILE_PATH")
        href = path_node.get("href")
        
        bands_in_file = []
        raster_display = df_container.find("Raster_Display")
        index_list = raster_display.find("Raster_Index_List")
        indices = []
        for idx_node in index_list.findall("Raster_Index"):
            b_id = idx_node.find("BAND_ID").text
            b_idx = int(idx_node.find("BAND_INDEX").text)
            indices.append((b_idx, b_id))
        indices.sort(key=lambda x: x[0])
        bands_in_file = [x[1] for x in indices]

        file_band_map[href] = bands_in_file

    output_data["Raster_Data"] = raster_data
    output_data["File_Band_Map"] = file_band_map

    # 4. Radiometric Data (Fixed for PNEO FWHM)
    bands_radiometry = {}
    meas_list = root.find(".//Radiometric_Data/Radiometric_Calibration/Instrument_Calibration/Band_Measurement_List")
    
    for measurement in meas_list:
        # Handle Standard/PHR structure
        measure_type = measurement.tag
        current_band_id = None
        measure_content = {}
        
        for child in measurement:
            if child.tag == "BAND_ID":
                current_band_id = child.text
            elif child.tag == "FWHM": # Flatten FWHM children directly
                for fwhm_child in child:
                    measure_content[fwhm_child.tag] = fwhm_child.text
            else:
                measure_content[child.tag] = child.text
        
        if current_band_id:
            if current_band_id not in bands_radiometry:
                bands_radiometry[current_band_id] = {}
            bands_radiometry[current_band_id][measure_type] = measure_content

    output_data["Radiometric_Calibration_By_Band"] = bands_radiometry

    # 5. Geometric Data
    geometric_positions = []
    use_area = root.find(".//Geometric_Data/Use_Area")
    for loc_val in use_area.findall("Located_Geometric_Values"):
        pos_data = {}
        loc_type = loc_val.find("LOCATION_TYPE")
        pos_data["LOCATION_TYPE"] = loc_type.text
        
        acq = loc_val.find("Acquisition_Angles")
        pos_data["Acquisition_Angles"] = {c.tag: c.text for c in acq}
        
        sol = loc_val.find("Solar_Incidences")
        pos_data["Solar_Incidences"] = {c.tag: c.text for c in sol}
        
        geometric_positions.append(pos_data)

    output_data["Geometric_Use_Area"] = geometric_positions
    return output_data

def interpolate_angle(values, dim_name, size):
    coords = {dim_name: [0, (size - 1) / 2, size - 1]}
    da = xr.DataArray(np.array(values, dtype=np.float32), dims=[dim_name], coords=coords)
    return interp(da, **{dim_name: Linear(np.arange(size), bounds="clip")})

def Level1_PNEO(
    product: Path,
    meta_data=False,
    subset_meta_data=[],
    chunks={"y": 1024, "x": 1024},
) -> xr.Dataset:
    """
    Read a Pleiades NEO Level-1 product as an xarray.Dataset.
    
    Pleiades NEO satellites provide high-resolution optical imagery with
    multiple spectral bands at ~50cm ground sample distance.

    Parameters
    ----------
    product : Path
        Path to the uncompressed Pleiades NEO product directory.
    meta_data : bool, optional
        If True, include full XML metadata in the Dataset attributes.
        Default is False.
    subset_meta_data : list[str], optional
        List of dot-separated paths to specific XML keys to include
        in attributes. Default is empty list.
    chunks : dict, optional
        Dask chunking configuration for lazy loading.
        Default is {"y": 1024, "x": 1024}.

    Returns
    -------
    xr.Dataset
        A georeferenced Xarray dataset with TOA reflectance and geometry layers.
    """
    if not product.exists():
        raise FileNotFoundError(f"Pleiades NEO product not found: {product}")

    prod = product
    while len(dirs := [d for d in prod.iterdir() if d.is_dir()]) == 1:
        prod = dirs[0]

    if len(dirs) >= 2:
        prod = next((d for d in dirs if "MS" in d.name), prod)
        
    xml_path = list(prod.glob("DIM*.XML"))[0]
    json_data = parse_dimap_data(xml_path)
    
    # 1. Load Raster Data
    file_band_map = json_data.get("File_Band_Map", {})
    data_arrays = []

    for filename, band_ids in file_band_map.items():
        jp2_candidates = list(prod.glob(filename))
            
        da = rioxarray.open_rasterio(jp2_candidates[0], chunks=chunks).rename(band='bands')
        da = da.drop_vars(["x", "y", "spatial_ref"], errors="ignore")
        da = da.assign_coords(bands=band_ids)
        data_arrays.append(da)
    ds = xr.concat(data_arrays, dim="bands")

    lat = np.array([float(v["LAT"]) for v in json_data["Dataset_Extent"]])
    lon = np.array([float(v["LON"]) for v in json_data["Dataset_Extent"]])
    rows = np.array([int(v["ROW"])-1 for v in json_data["Dataset_Extent"]])
    cols = np.array([int(v["COL"])-1 for v in json_data["Dataset_Extent"]])
    
    dtype = [('row', int), ('col', int), ('lat', float), ('lon', float)]
    structured_data = np.array(list(zip(rows, cols, lat, lon)), dtype=dtype)
    sorted_data = np.sort(structured_data, order=['row', 'col'])
    
    rows_sorted = sorted_data['row'].reshape((2, 2))
    cols_sorted = sorted_data['col'].reshape((2, 2))
    lat2D = sorted_data['lat'].reshape((2, 2))
    lon2D = sorted_data['lon'].reshape((2, 2))

    rows_coord = rows_sorted[:, 0]
    cols_coord = cols_sorted[0, :]

    longitude = xr.DataArray(
        np.array(lon2D, dtype='float32'), dims=["y_", "x_"], name="longitude",
        coords={"y_": rows_coord, "x_": cols_coord}
    )
    latitude = xr.DataArray(
        np.array(lat2D, dtype='float32'), dims=["y_", "x_"], name="latitude",
        coords={"y_": rows_coord, "x_": cols_coord}
    )

    y_coords = xr.DataArray(np.arange(ds.y.size), dims=['y'])
    x_coords = xr.DataArray(np.arange(ds.x.size), dims=['x'])

    interpolated_lon = interp(
        longitude,
        y_=Linear(y_coords, bounds="clip"),
        x_=Linear(x_coords, bounds="clip"),
    ).transpose('y', 'x')
    interpolated_lat = interp(
        latitude,
        y_=Linear(y_coords, bounds="clip"),
        x_=Linear(x_coords, bounds="clip"),
    ).transpose('y', 'x')

    angles_list = list(json_data["Geometric_Use_Area"])
    sza = np.array([90.0 - float(v["Solar_Incidences"]["SUN_ELEVATION"]) for v in angles_list])
    saa = np.array([float(v["Solar_Incidences"]["SUN_AZIMUTH"]) for v in angles_list])
    vza = np.array([float(v["Acquisition_Angles"]["VIEWING_ANGLE"]) for v in angles_list])
    vaa = np.array([float(v["Acquisition_Angles"]["AZIMUTH_ANGLE"]) for v in angles_list])

    ones_x = xr.DataArray(np.ones(ds.x.size, dtype=np.float32), dims=["x"], coords={"x": np.arange(ds.x.size)})
    
    interpolated_sza = interpolate_angle(sza, "y", ds.y.size) * ones_x
    interpolated_vza = interpolate_angle(vza, "y", ds.y.size) * ones_x
    interpolated_saa = interpolate_angle(saa, "y", ds.y.size) * ones_x
    interpolated_vaa = interpolate_angle(vaa, "y", ds.y.size) * ones_x

    band_infos = json_data["Radiometric_Calibration_By_Band"]
    current_bands = ds.bands.values.tolist()
    
    g_list, b_list, e_list, mins, maxs = [], [], [], [], []
    
    for b in current_bands:
        info = band_infos[b]
        g_list.append(float(info["Band_Reflectance"]["GAIN"]))
        b_list.append(float(info["Band_Reflectance"]["BIAS"]))
        e_list.append(float(info["Band_Solar_Irradiance"]["VALUE"]))
        mins.append(float(info["Band_Spectral_Range"]["MIN"]))
        maxs.append(float(info["Band_Spectral_Range"]["MAX"]))

    np_mins = np.array(mins)
    np_maxs = np.array(maxs)

    da_kwargs = {"dims": ["bands"], "coords": {"bands": ds.bands}}

    gains = xr.DataArray(np.array(g_list, dtype=np.float32), **da_kwargs)
    biais = xr.DataArray(np.array(b_list, dtype=np.float32), **da_kwargs)
    
    cwav = xr.DataArray(((np_maxs + np_mins) / 2).astype(np.float32), **da_kwargs)

    attrs = json_data["Geodetic_CRS"] | json_data["Identification"] | json_data["Source_ID"]
    attrs["Special_Values"] = json_data["Raster_Data"].get("Special_Values", [])

    mus = np.cos(np.radians(interpolated_sza))
    rho_toa = (ds / gains + biais) / mus
    ds_final = xr.Dataset(
        {
            str(n.lon): interpolated_lon,
            str(n.lat): interpolated_lat,
            str(n.sza): interpolated_sza,
            str(n.saa): interpolated_saa,
            str(n.vza): interpolated_vza,
            str(n.vaa): interpolated_vaa,
            str(n.cwav): cwav,
            str(n.rtoa): rho_toa,
        },
        coords={"bands": ds.bands},
        attrs=attrs,
    ).chunk(chunks)

    for v in [n.rtoa, n.lon, n.lat, n.sza, n.saa, n.vza, n.vaa, n.cwav, n.wav]:
        if str(v) in ds_final:
            ds_final[str(v)].attrs = v.attrs
            
    combined_str = f"{ds_final.attrs["IMAGING_DATE"]} {ds_final.attrs["IMAGING_TIME"]}".removesuffix("Z")

    ds_final.attrs[n.datetime] = combined_str
    del ds_final.attrs["IMAGING_DATE"], ds_final.attrs["IMAGING_TIME"]
    
    if meta_data or subset_meta_data != []:
        xmls = list(prod.glob("*XML"))

        xmls = [d for d in xmls if "NED.XML" not in d.name]

        dict_xml = {
            xmls[i].name : parse(open(xmls[i]).read()) for i in range(len(xmls))
        }
        
        meta_data_dict = {}
        if subset_meta_data != []:
            for dict_key in dict_xml:
                meta_data_dict = meta_data_dict | iter_through_subset(dict_xml[dict_key], subset_meta_data)
        else :
            meta_data_dict = dict_xml

        ds_final.attrs = ds_final.attrs | meta_data_dict

    # set srf getter (PHR1A, PHR1B, PNEO3, PNEO4)
    ds_final.attrs.update({'_srf_getter': 'eoread.pleiades.read_srf_pleiades'})
    srf_args = ds_final.attrs['MISSION'] + ds_final.attrs['MISSION_INDEX']
    ds_final.attrs.update({'_srf_getter_arg': srf_args})

    return ds_final.sel(bands=['B', 'G', 'R', 'NIR'])

def get_sample() -> Path:
    return uncompress_decorator()(download_url)(
        "https://earth.esa.int/eogateway/ftp/missions/sample-data/third-party-missions/pleiades-neo/WO_000194880_1_1_SAL24178541-1_ACQ_PNEO4_03432708887687.zip",
        getdir("DIR_SAMPLES") / "PLEIADES-NEO",
    )