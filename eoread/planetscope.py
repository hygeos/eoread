from pathlib import Path
from core.network.download import download_url
from core.env import getdir
from core.uncompress import uncompress_decorator
import xarray as xr
import rioxarray
import rasterio
import xml.etree.ElementTree as ET
import numpy as np
from core.interpolate import interp, Linear 
from core.geo import n
from core.tools import only
from datetime import datetime
from xmltodict import parse
from jrc_rayleigh_processor.utils import iter_through_subset

user_guide = "https://assets.planet.com/docs/Planet_PSScene_Imagery_Product_Spec_June_2021.pdf"

def xml_to_dict(el):
    """
    Recursively converts an XML element and its children into a dictionary.
    Handles namespaces, attributes, and repeating child nodes.
    """
    res = {}
    
    # Add attributes (prefixed with @)
    for name, value in el.attrib.items():
        attr_name = name.split('}')[-1] if '}' in name else name
        res[f"@{attr_name}"] = value

    # Process children
    for child in el:
        child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        child_value = xml_to_dict(child)
        
        if child_tag not in res:
            res[child_tag] = child_value
        else:
            # If tag exists, convert to list (crucial for bandSpecificMetadata)
            if not isinstance(res[child_tag], list):
                res[child_tag] = [res[child_tag]]
            res[child_tag].append(child_value)

    # Return text if no children/attributes, otherwise combine
    if not res:
        return el.text.strip() if el.text else None
    
    if el.text and el.text.strip():
        if len(res) == 0: 
            return el.text.strip()
        else:
            res["#text"] = el.text.strip()
            
    return res

def parse_planet_xml(file_path):
    """Parses the XML file and returns a dictionary."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Wrap in the root tag name
        root_tag = root.tag.split('}')[-1]
        return {root_tag: xml_to_dict(root)}
    
    except FileNotFoundError:
        return {"error": "File not found."}
    except ET.ParseError as e:
        return {"error": f"XML Parse Error: {e}"}

def Level1_Planetscope(
    product: Path,
    meta_data=False,
    subset_meta_data=[],
    chunks={"x": 1024, "y": 1024},
) -> xr.Dataset:
    """
    Read a PlanetScope Level-1 product as an xarray.Dataset.
    
    PlanetScope satellites provide high-resolution optical imagery with
    8 spectral bands at ~3m ground sample distance.

    Parameters
    ----------
    product : Path
        Path to the uncompressed PlanetScope product directory.
    meta_data : bool, optional
        If True, include full XML metadata in the Dataset attributes.
        Default is False.
    subset_meta_data : list[str], optional
        List of dot-separated paths to specific XML keys to include
        in attributes. Default is empty list.
    chunks : dict, optional
        Dask chunking configuration for lazy loading.
        Default is {"x": 1024, "y": 1024}.

    Returns
    -------
    xr.Dataset
        A georeferenced Xarray dataset with TOA reflectance and geometry layers.
    """
    if not product.exists():
        raise FileNotFoundError(f"PlanetScope product not found: {product}")

    file_name_tif = "**/*8b.tif"
    file_tif = only(product.glob(file_name_tif))

    file_name_dim = "**/*.xml"
    file_xml = only(product.glob(file_name_dim))

    da = rioxarray.open_rasterio(file_tif, chunks=chunks).drop_vars("spatial_ref").rename(band='bands')

    da = da.assign_coords(bands=list(da.attrs['long_name']))

    shape = (da.y.size, da.x.size)

    metadata = parse_planet_xml(file_xml)

    bands_cwav = np.array([443, 490, 531, 565, 610, 665, 705, 865], dtype='float32')

    bands_info = metadata["EarthObservation"]["resultOf"]["EarthObservationResult"][
        "bandSpecificMetadata"
    ]
    bands_info_ordered = [{} for _ in range(len(bands_info))]
    for band in bands_info:
        band_id = int(band["bandNumber"]) - 1
        bands_info_ordered[band_id] = band

    ratio_rtoa = []
    ratio_ltoa = []
    for band in bands_info_ordered:
        ratio_rtoa.append(float(band["reflectanceCoefficient"]))
        ratio_ltoa.append(float(band["radiometricScaleFactor"]))
        
    da_ratio_rtoa = xr.DataArray(
        np.array(ratio_rtoa, dtype='float32'), dims=["bands"], name="ratio_rtoa"
    )

    # TODO: check whether we should divide by cos(sza)
    da_rtoa = da * da_ratio_rtoa

    angles = metadata["EarthObservation"]["using"]["EarthObservationEquipment"][
        "acquisitionParameters"
    ]["Acquisition"]

    saa = np.full(shape, float(angles["illuminationAzimuthAngle"]["#text"]))
    sza = np.full(shape, 90.0 - float(angles["illuminationElevationAngle"]["#text"]))
    vaa = np.full(shape, float(angles["azimuthAngle"]["#text"]))
    vza = np.full(shape, float(angles["spaceCraftViewAngle"]["#text"]))

    acquisition_date = metadata["EarthObservation"]["metaDataProperty"]["EarthObservationMetaData"]["downlinkedTo"]["DownlinkInformation"]["acquisitionDate"]
    product_type = metadata["EarthObservation"]["metaDataProperty"]["EarthObservationMetaData"]["productType"]
    sensor_info = metadata["EarthObservation"]["using"]["EarthObservationEquipment"][
        "platform"
    ]["Platform"]

    # Use RPC (Rational Polynomial Coefficients) for non-linear geolocation
    # instead of simple linear interpolation of corner coordinates
    rpc_file = list(product.glob("**/*_RPC.TXT"))
    if rpc_file:
        # Parse RPC parameters from text file
        rpc_dict = {}
        for line in rpc_file[0].read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                rpc_dict[key.strip()] = float(value.strip())
        
        # Build rasterio RPC object and use RPCTransformer
        from rasterio.rpc import RPC
        rpc = RPC(
            line_off=rpc_dict['LINE_OFF'],
            line_scale=rpc_dict['LINE_SCALE'],
            line_num_coeff=[rpc_dict[f'LINE_NUM_COEFF_{i}'] for i in range(1, 21)],
            line_den_coeff=[rpc_dict[f'LINE_DEN_COEFF_{i}'] for i in range(1, 21)],
            samp_off=rpc_dict['SAMP_OFF'],
            samp_scale=rpc_dict['SAMP_SCALE'],
            samp_num_coeff=[rpc_dict[f'SAMP_NUM_COEFF_{i}'] for i in range(1, 21)],
            samp_den_coeff=[rpc_dict[f'SAMP_DEN_COEFF_{i}'] for i in range(1, 21)],
            lat_off=rpc_dict['LAT_OFF'],
            lat_scale=rpc_dict['LAT_SCALE'],
            long_off=rpc_dict['LONG_OFF'],
            long_scale=rpc_dict['LONG_SCALE'],
            height_off=rpc_dict['HEIGHT_OFF'],
            height_scale=rpc_dict['HEIGHT_SCALE'],
        )
        
        transformer = rasterio.transform.RPCTransformer(rpc)
        
        # Create pixel coordinate grids (line = y, sample = x)
        y_indices = np.arange(da.y.size, dtype=np.float64)
        x_indices = np.arange(da.x.size, dtype=np.float64)
        line_grid, sample_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Transform pixel coordinates to lat/lon using RPC
        # Note: transformer.xy returns (lon, lat) order, flattened arrays
        lon_array, lat_array = transformer.xy(line_grid, sample_grid, zs=0.0)
        lon_array = np.asarray(lon_array, dtype=np.float32).reshape(line_grid.shape)
        lat_array = np.asarray(lat_array, dtype=np.float32).reshape(sample_grid.shape)
        
        interpolated_lon = xr.DataArray(
            lon_array,
            dims=['y', 'x'],
        )
        interpolated_lat = xr.DataArray(
            lat_array,
            dims=['y', 'x'],
        )
    else:
        # Fallback to linear interpolation of corner coordinates if no RPC file
        footprint = metadata["EarthObservation"]["target"]["Footprint"]["geographicLocation"]
        
        lons = np.array([
            [float(footprint["topLeft"]["longitude"]), float(footprint["topRight"]["longitude"])],
            [float(footprint["bottomLeft"]["longitude"]), float(footprint["bottomRight"]["longitude"])]
        ], dtype=np.float32)
        
        lats = np.array([
            [float(footprint["topLeft"]["latitude"]), float(footprint["topRight"]["latitude"])],
            [float(footprint["bottomLeft"]["latitude"]), float(footprint["bottomRight"]["latitude"])]
        ], dtype=np.float32)

        y_coords = [0, da.y.size - 1]
        x_coords = [0, da.x.size - 1]

        da_lon_base = xr.DataArray(lons, dims=["y_", "x_"], coords={"y_": y_coords, "x_": x_coords})
        da_lat_base = xr.DataArray(lats, dims=["y_", "x_"], coords={"y_": y_coords, "x_": x_coords})

        y_coords = xr.DataArray(np.arange(da.y.size), dims=['y'])
        x_coords = xr.DataArray(np.arange(da.x.size), dims=['x'])
        interpolated_lon = interp(
            da_lon_base,
            y_=Linear(y_coords, bounds="clip"),
            x_=Linear(x_coords, bounds="clip"),
        ).transpose('y', 'x')
        interpolated_lat = interp(
            da_lat_base,
            y_=Linear(y_coords, bounds="clip"),
            x_=Linear(x_coords, bounds="clip"),
        ).transpose('y', 'x')

    ds = xr.Dataset(
        {
            str(n.rtoa): da_rtoa.astype(n.rtoa.dtype),#/np.cos(np.radians(sza)),
            str(n.saa): (("y", "x"), saa.astype(n.saa.dtype)),
            str(n.sza): (("y", "x"), sza.astype(n.sza.dtype)),
            str(n.vaa): (("y", "x"), vaa.astype(n.vaa.dtype)),
            str(n.vza): (("y", "x"), vza.astype(n.vza.dtype)),
            str(n.cwav): (("bands",), bands_cwav),
            str(n.lon): interpolated_lon.astype(n.lon.dtype),
            str(n.lat): interpolated_lat.astype(n.lat.dtype),
        },
        attrs = sensor_info | {"product_type": product_type, "acquisition_date": acquisition_date},
    ).chunk(chunks)

    for v in [n.rtoa, n.lon, n.lat, n.sza, n.saa, n.vza, n.vaa, n.cwav, n.wav]:
        if str(v) in ds:
            ds[str(v)].attrs = v.attrs
    
    ds.attrs[n.datetime] = ds.attrs["acquisition_date"]
    
    ds.attrs["acquisition_date"]
    
    ds.attrs["user_guide"] = user_guide
    
    if meta_data or subset_meta_data != []:
        dict_ = parse(open(file_xml).read())
        
        ds.attrs = ds.attrs | iter_through_subset(dict_, subset_meta_data)
    
    ds.attrs.update({'_srf_getter': 'eoread.planetscope.read_srf_superdove'})

    return ds

def read_srf_superdove(srf_file: Path | None = None) -> xr.Dataset:
    import pandas as pd

    if srf_file is None:
        url = "https://github.com/hygeos/eoread/releases/download/root/Superdove_SRF.csv"
        srf_file = download_url(url, getdir("DIR_STATIC") / "srf" / "planetscope")
    
    df = pd.read_csv(srf_file, sep=';')
    
    wavelength = df['Wavelength (nm)']
    
    # band_names = ["Coastal-Blue", "Blue", "Green_i", "Green_ii", "Yellow", "Red", "Red-edge", "NIR"]
    band_names = ['441', '490', '531', '565', '610', '665', '705', '865']
    
    ds = xr.Dataset()
    for band in band_names:
        ds[band] = xr.DataArray(df[band].values, dims=['wavelength'])
    
    ds = ds.assign_coords(wavelength=wavelength.values)

    # Remove negative/invalid values
    ds = ds.where(ds>0, 0)

    # Set units
    ds.wavelength.attrs.update(units='nm')
    
    return ds


def get_sample() -> Path:
    return uncompress_decorator()(download_url)(
        "https://earth.esa.int/eogateway/ftp/missions/sample-data/third-party-missions/planetscope/PSScene_Basic_Analytic_8b_udm2.zip",
        getdir("DIR_SAMPLES") / "PLANETSCOPE",
    )