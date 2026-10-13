from pathlib import Path
from core.network.download import download_url
from core.env import getdir
from core.uncompress import uncompress_decorator
import xarray as xr
import rioxarray
from core.interpolate import interp, Linear
from core.geo import n
import numpy as np
from jrc_rayleigh_processor.utils import iter_through_subset
import xml.etree.ElementTree as ET
from xmltodict import parse


band_names = ['BLUE', 'GREEN', 'RED', 'NIR']

def parse_dimap_data(xml_file_path):
    """
    Parses a DIMAP XML file and extracts specific dataset, raster, 
    radiometric, and geometric information into a dictionary.
    AI
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    output_data = {}

    # ---------------------------------------------------------
    # 1. Dataset Extent (Vertices)
    # ---------------------------------------------------------
    # Path: Dimap_Document -> Dataset_Content -> Dataset_Extent -> Vertex
    vertices = []
    dataset_extent_node = root.find(".//Dataset_Content/Dataset_Extent")
    for vertex in dataset_extent_node.findall("Vertex"):
        vertices.append({
            "LON": vertex.find("LON").text,
            "LAT": vertex.find("LAT").text,
            "COL": vertex.find("COL").text,
            "ROW": vertex.find("ROW").text
        })
    output_data["Dataset_Extent"] = vertices

    
    # ---------------------------------------------------------
    # 2. Delivery Identification
    # ---------------------------------------------------------
    ident_node = root.find(".//Product_Information/Delivery_Identification")
    output_data["Identification"] = {child.tag: child.text for child in ident_node}
    del output_data["Identification"]["Order_Identification"]
    
    source_node = root.find(".//Dataset_Sources/Source_Identification/Strip_Source")
    output_data["Source_ID"] = {child.tag: child.text for child in source_node}
    

    # ---------------------------------------------------------
    # 3. Geodetic CRS
    # ---------------------------------------------------------
    # Path: Dimap_Document -> Coordinate_Reference_System -> Geodetic_CRS
    crs_node = root.find(".//Coordinate_Reference_System/Geodetic_CRS")
    output_data["Geodetic_CRS"] = {child.tag: child.text for child in crs_node}

    # ---------------------------------------------------------
    # 4. Raster Data (Dimensions & Display)
    # ---------------------------------------------------------
    raster_data = {}
    
    # Raster Dimensions
    dims_node = root.find(".//Raster_Data/Raster_Dimensions")
    raster_data["Raster_Dimensions"] = {child.tag: child.text for child in dims_node}
        
    # Raster Display (Band Order & Special Values)
    display_node = root.find(".//Raster_Data/Raster_Display")
    display_info = {}
    
    # Band Display Order
    order_node = display_node.find("Band_Display_Order")
    
    display_info["Band_Display_Order"] = {child.tag: child.text for child in order_node}
        
    # Special Values (Multiple entries possible)
    special_values = []
    for sv_node in display_node.findall("Special_Value"):
        special_values.append({child.tag: child.text for child in sv_node})
    display_info["Special_Values"] = special_values
    
    raster_data["Raster_Display"] = display_info
        
    output_data["Raster_Data"] = raster_data

    # ---------------------------------------------------------
    # 5. Radiometric Data (Combined by Band)
    # ---------------------------------------------------------
    # Path: Dimap_Document -> Radiometric_Data -> Radiometric_Calibration -> Instrument_Calibration -> Band_Measurement_List
    # We iterate over the list and group items by their 'BAND_ID'
    bands_radiometry = {}
    
    meas_list = root.find(".//Radiometric_Data/Radiometric_Calibration/Instrument_Calibration/Band_Measurement_List")
    
    for measurement in meas_list:
        # The tag name (e.g., Band_Spectral_Range, Band_Radiance, etc.)
        measure_type = measurement.tag
        
        current_band_id = None
        measure_content = {}
        
        for child in measurement:
            if child.tag == "BAND_ID":
                current_band_id = child.text
            else:
                measure_content[child.tag] = child.text
        
        if current_band_id:
            if current_band_id not in bands_radiometry:
                bands_radiometry[current_band_id] = {}
            bands_radiometry[current_band_id][measure_type] = measure_content

    output_data["Radiometric_Calibration_By_Band"] = bands_radiometry

    # ---------------------------------------------------------
    # 6. Geometric Data (Use Area)
    # ---------------------------------------------------------
    # Path: Dimap_Document -> Geometric_Data -> Use_Area
    # Contains multiple 'Located_Geometric_Values'
    geometric_positions = []
    use_area = root.find(".//Geometric_Data/Use_Area")
    for loc_val in use_area.findall("Located_Geometric_Values"):
        pos_data = {}
        
        # Position Name (Top Center, etc.)
        loc_type = loc_val.find("LOCATION_TYPE")
        pos_data["LOCATION_TYPE"] = loc_type.text
        
        # Angles
        acq_angles = loc_val.find("Acquisition_Angles")
        pos_data["Acquisition_Angles"] = {child.tag: child.text for child in acq_angles}
            
        # Solar Incidences
        sol_inc = loc_val.find("Solar_Incidences")
        pos_data["Solar_Incidences"] = {child.tag: child.text for child in sol_inc}
            
        geometric_positions.append(pos_data)

    output_data["Geometric_Use_Area"] = geometric_positions

    return output_data

def interpolate_angle(values, dim_name, size):
    """
    Creates a DataArray from 3 points (start, mid, end) 
    and interpolates to the full dimension size.
    """
    # Define the 3 control points: [0, middle_index, last_index]
    coords = {dim_name: [0, (size - 1) / 2, size - 1]}
    
    da = xr.DataArray(
        np.array(values, dtype=np.float32), 
        dims=[dim_name], 
        coords=coords
    )
    
    # Interpolate across the full range of the dimension
    return interp(
        da, 
        **{dim_name: Linear(np.arange(size), bounds="clip")}
    )
    

def Level1_Pleiades(
    product: Path,
    meta_data=False,
    subset_meta_data=[],
    chunks={"y": 1024, "x": 1024},
) -> xr.Dataset:
    """
    Read a Pleiades Level-1 product as an xarray.Dataset.
    
    Pleiades satellites (Pleiades-1A/1B) provide high-resolution optical imagery
    with multiple spectral bands at ~50cm ground sample distance.

    Parameters
    ----------
    product : Path
        Path to the uncompressed Pleiades product directory.
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
        raise FileNotFoundError(f"Pleiades product not found: {product}")

    prod = product
    while len(dirs := [d for d in prod.iterdir() if d.is_dir()]) == 1:
        prod = dirs[0]

    if len(dirs) >= 2:
        prod = next((d for d in dirs if "MS" in d.name), prod)
    
    data_path = list(prod.glob("*R1C1.JP2"))[0]
    
    da = rioxarray.open_rasterio(data_path, chunks=chunks).rename(band="bands")
    da = da.drop_vars(["x", "y", "spatial_ref"], errors="ignore")
    json_path = list(prod.glob("DIM*.XML"))[0]
    json = parse_dimap_data(json_path)

    lat = np.array([float(v["LAT"]) for v in json["Dataset_Extent"]])
    lon = np.array([float(v["LON"]) for v in json["Dataset_Extent"]])
    rows = np.array([int(v["ROW"])-1 for v in json["Dataset_Extent"]])
    cols = np.array([int(v["COL"])-1 for v in json["Dataset_Extent"]])
    reorder = np.array([0, 1, 3, 2])

    lat2D = lat[reorder].reshape((2,2))
    lon2D = lon[reorder].reshape((2,2))
    rows2D = rows[reorder].reshape((2,2))
    cols2D = cols[reorder].reshape((2,2))

    rows2D = rows2D[:, 0]
    cols2D = cols2D[0, :]

    longitude = xr.DataArray(
        np.array(lon2D).astype(np.float32), dims=["y_", "x_"], name="longitude",
        coords={"y_": rows2D, "x_": cols2D}
    )

    latitude = xr.DataArray(
        np.array(lat2D).astype(np.float32), dims=["y_", "x_"], name="latitude",
        coords={"y_": rows2D, "x_": cols2D}
    )

    y_coords = xr.DataArray(np.arange(da.y.size), dims=['y'])
    x_coords = xr.DataArray(np.arange(da.x.size), dims=['x'])
    
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

    attrs = json["Geodetic_CRS"] | json["Identification"] | json["Source_ID"]

    special_values = json["Raster_Data"]["Raster_Display"]["Special_Values"]

    attrs = attrs | {"Special_Values": special_values}

    band_infos = json["Radiometric_Calibration_By_Band"]
    bands = json["Raster_Data"]["Raster_Display"]["Band_Display_Order"]

    for k, v in bands.items():
        band_infos[v]["name"] = k

    angles_json =list(json["Geometric_Use_Area"])

    sza = np.array([90.0 - float(v["Solar_Incidences"]["SUN_ELEVATION"]) for v in angles_json])
    saa = np.array([float(v["Solar_Incidences"]["SUN_AZIMUTH"]) for v in angles_json])
    vza = np.array([float(v["Acquisition_Angles"]["VIEWING_ANGLE"]) for v in angles_json])
    vaa = np.array([float(v["Acquisition_Angles"]["AZIMUTH_ANGLE"]) for v in angles_json])

    ones_x = xr.DataArray(np.ones(da.x.size), dims=["x"], coords={"x": np.arange(da.x.size)})

    interpolated_sza = interpolate_angle(sza, "y", da.y.size) * ones_x
    interpolated_vza = interpolate_angle(vza, "y", da.y.size) * ones_x
    interpolated_saa = interpolate_angle(saa, "y", da.y.size) * ones_x
    interpolated_vaa = interpolate_angle(vaa, "y", da.y.size) * ones_x


    band_data = {
        f"B{i}": band_infos[f"B{i}"] for i in range(da.bands.size)
    }

    g_list = [float(band_data[b]["Band_Reflectance"]["GAIN"]) for b in band_data]
    b_list = [float(band_data[b]["Band_Reflectance"]["BIAS"]) for b in band_data]

    mins = np.array([float(band_data[b]["Band_Spectral_Range"]["MIN"]) for b in band_data]) * 1000
    maxs = np.array([float(band_data[b]["Band_Spectral_Range"]["MAX"]) for b in band_data]) * 1000

    da_kwargs = {"dims": ["bands"], "coords": {"bands": da.bands}}

    gains = xr.DataArray(np.array(g_list, dtype=np.float32), **da_kwargs)
    bias = xr.DataArray(np.array(b_list, dtype=np.float32), **da_kwargs)

    cwav = xr.DataArray(((maxs + mins) / 2).astype(np.float32), **da_kwargs)
    
    mus = np.cos(np.radians(interpolated_sza))
    rho_toa = (da / gains + bias) / mus

    ds_final = (
        xr.Dataset(
            {
                str(n.lon): interpolated_lon,
                str(n.lat): interpolated_lat,
                str(n.sza): interpolated_sza,
                str(n.saa): interpolated_saa,
                str(n.vza): interpolated_vza,
                str(n.vaa): interpolated_vaa,
                str(n.cwav): (("bands",), np.array(cwav, dtype=np.float32)),
                str(n.rtoa): rho_toa,
            },
            attrs=attrs,
        )
        .chunk(chunks)
        .assign_coords(bands=band_names)
    )
    

    for v in [n.rtoa, n.lon, n.lat, n.sza, n.saa, n.vza, n.vaa, n.cwav, n.wav]:
        if str(v) in ds_final:
            ds_final[str(v)].attrs = v.attrs
    
    combined_str = f"{ds_final.attrs['IMAGING_DATE']} {ds_final.attrs['IMAGING_TIME']}".removesuffix("Z")

    ds_final.attrs[n.datetime] = combined_str
    del ds_final.attrs["IMAGING_DATE"], ds_final.attrs["IMAGING_TIME"]
    
    if meta_data or subset_meta_data != [] :
        xmls = list(prod.glob("*XML"))
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
    srf_args = ds_final.attrs['INSTRUMENT'] + ds_final.attrs['INSTRUMENT_INDEX']
    ds_final.attrs.update({'_srf_getter_arg': srf_args})
    
    return ds_final


def get_sample() -> Path:
    return uncompress_decorator()(download_url)(
        "https://earth.esa.int/eogateway/ftp/missions/sample-data/third-party-missions/pleiades/Polygon1_SO24012539-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298.zip",
        getdir("DIR_SAMPLES") / "PLEIADES",
    )


def read_srf_pleiades(
    sensor_name: str,
    srf_file: Path | None = None,
    panchromatic: bool = False,
) -> xr.Dataset:
    """
    Read SRF (Spectral Response Function) data for pleiades, pleiades-neo, spot
    
    Parameters
    ----------
    sensor_name : str
        Name of the sensor (e.g., "pleiades", "spot", "worldview").
    srf_file : Path, optional
        Path to the Excel file containing SRF data.
    panchromatic : bool, optional
        Whether to return panchromatic SRF data.
    
    Returns
    -------
    xr.Dataset
        Dataset containing SRF data with one variable per spectral band.
    """
    import pandas as pd
    if srf_file is None:
        url = "https://github.com/hygeos/eoread/releases/download/root/SpectralValues_Normalized-PHR1A_PHR1B_SPOT6_SPOT7_PNEO3-4.xlsx"
        srf_file = download_url(url, getdir("DIR_STATIC") / "srf" / "pleiades")
    
    # Load Excel file
    df = pd.read_excel(srf_file, sheet_name=0)
    
    # Create xarray dataset
    ds = xr.Dataset()
    
    # Extract wavelength and response data
    assert 'Lambda(mm)' in df.columns
    wavelength = xr.DataArray(
        df['Lambda(mm)'].values,
        dims=['wavelength'],
        name='wavelength'
    )
    
    # Extract response data for all bands (excluding wavelength column)
    band_cols = [col for col in df.columns if col != 'Lambda(mm)']
    for band in band_cols:
        response = xr.DataArray(
            df[band].values,
            dims=['wavelength'],
            name=band,
            attrs={'sensor': sensor_name}
        )
        ds[band] = response
    
    ds = ds.assign_coords(wavelength=wavelength*1000)
    ds.wavelength.attrs.update(units='nm')

    
    # Add metadata
    ds.attrs['sensor'] = sensor_name
    ds.attrs['source_file'] = str(srf_file)
    
    if not panchromatic:
        return ds[[x for x in ds if x != 'PAN']]
    else:
        return ds