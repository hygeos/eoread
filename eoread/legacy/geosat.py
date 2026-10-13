from pathlib import Path
from core.network.download import download_url
from core.env import getdir
from core.uncompress import uncompress_decorator
from scipy.interpolate import RectBivariateSpline
import xarray as xr
import rioxarray
import rasterio
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from core.geo import n
from typing import Literal
from xmltodict import parse
from core.tools import only
from jrc_rayleigh_processor.utils import iter_through_subset


def parse_dimap_metadata(file_input):
    """
    Parses a DIMAP XML file.
    Accepts a file path (string or Path object).
    """
    try:
        # Use ET.parse() for files instead of ET.fromstring()
        tree = ET.parse(file_input)
        root = tree.getroot()
    except ET.ParseError as e:
        return {"error": f"Failed to parse XML: {e}"}
    except FileNotFoundError:
        return {"error": f"File not found: {file_input}"}

    # Helper to find text safely
    def get_text(path, default=None):
        node = root.find(path)
        return node.text if node is not None else default

    # Helper to find text safely within a parent element
    def get_text_from(parent, path, default=None):
        node = parent.find(path)
        return node.text if node is not None else default

    # Helper for Quality Parameters (which repeat)
    def get_quality_params(parent_path):
        params = {}
        target = root.find(parent_path)
        if target is not None:
            for param in target.findall("Quality_Parameter"):
                code_node = param.find("QUALITY_PARAMETER_CODE")
                val_node = param.find("QUALITY_PARAMETER_VALUE")
                if code_node is not None and val_node is not None:
                    params[code_node.text] = val_node.text
        return params

    metadata = {
        "identification": {
            "dataset_name": get_text(".//Dataset_Id/DATASET_NAME"),
            "product_type": get_text(".//Production/PRODUCT_TYPE"),
            "production_date": get_text(".//Production/DATASET_PRODUCTION_DATE"),
            "copyright": get_text(".//Dataset_Id/COPYRIGHT"),
        },
        "geospatial": {
            "crs_code": get_text(".//Horizontal_CS/HORIZONTAL_CS_CODE"),
            "crs_name": get_text(".//Horizontal_CS/HORIZONTAL_CS_NAME"),
            "vertices": [
                {
                    "lat": get_text_from(v, "FRAME_LAT"),
                    "lon": get_text_from(v, "FRAME_LON"),
                    "row": get_text_from(v, "FRAME_ROW"),
                    "col": get_text_from(v, "FRAME_COL"),
                }
                for v in root.findall(".//Dataset_Frame/Vertex")
            ],
        },
        "acquisition": {
            "mission": get_text(".//Scene_Source/MISSION"),
            "instrument": get_text(".//Scene_Source/INSTRUMENT"),
            "imaging_date": get_text(".//Scene_Source/IMAGING_DATE"),
            "imaging_time": get_text(".//Scene_Source/IMAGING_TIME"),
            "SUN_AZIMUTH": float(get_text(".//Scene_Source/SUN_AZIMUTH")),
            "SUN_ELEVATION": float(get_text(".//Scene_Source/SUN_ELEVATION")),
            "incidence_angle": float(get_text(".//Scene_Source/INCIDENCE_ANGLE")),
            "VIEWING_ANGLE": float(get_text(".//Scene_Source/VIEWING_ANGLE")),
            "resolution": get_text(".//Scene_Source/THEORETICAL_RESOLUTION"),
        },
        "spectral_bands": [],
    }

    # Extract detailed band info (Gain/Bias/Stats)
    for band in root.findall(".//Spectral_Band_Info"):
        idx = get_text_from(band, "BAND_INDEX")

        band_data = {
            "index": idx,
            "description": get_text_from(band, "BAND_DESCRIPTION"),
            "physical_gain": get_text_from(band, "PHYSICAL_GAIN"),
            "physical_bias": get_text_from(band, "PHYSICAL_BIAS"),
            "unit": get_text_from(band, "PHYSICAL_UNIT"),
        }
        metadata["spectral_bands"].append(band_data)

    return metadata


def Level1_GEOSAT(
    product: Path,
    meta_data=False,
    subset_meta_data=[],
    chunks={"x": 1024, "y": 1024},
):
    """
    Read a GEOSAT Level-1 product as an xarray.Dataset.
    
    GEOSAT satellites provide high-resolution optical imagery in multiple
    spectral bands. This reader handles both GEOSAT-1 (DE-1) and GEOSAT-2 (DE-2)
    products with different georeferencing schemes.

    Parameters
    ----------
    product : Path
        Path to the uncompressed Geosat product directory.
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
        A georeferenced Xarray dataset with TOA Radiance and geometry layers.
    """
    if not product.exists():
        raise FileNotFoundError(f"GEOSAT product not found: {product}")

    file_name_tif = "*.tif"
    file_tif = only(product.glob(file_name_tif))

    da = rioxarray.open_rasterio(
        file_tif, chunks=chunks
    ).drop_vars("spatial_ref")

    da = da.rename(band="bands")

    # Derive lat/lon from the GeoTIFF's native georeferencing.
    # GEOSAT products have different georeferencing schemes:
    # - GEOSAT-1 (DE-1): Identity transform + 25 GCPs (5x5 grid) defining distorted pushbroom geometry.
    #   Use GCP-based spline interpolation
    # - GEOSAT-2 (DE-2): Proper UTM affine transform (e.g. EPSG:32631), no GCPs.
    #   Use affine transform + CRS transformation (UTM → WGS84) for lat/lon.
    
    file_name_dim = "*.dim"
    file_dim = only(product.glob(file_name_dim))
    attrs = parse_dimap_metadata(str(file_dim))
    
    # Read GCPs and transform from the GeoTIFF
    with rasterio.open(file_tif) as src:
        gcps, gcps_crs = src.gcps
        src_transform = src.transform
        src_crs = src.crs
    
    if gcps:
        # GEOSAT-1: Use GCP-based spline interpolation
        # Extract GCP pixel coordinates (row/col) and lat/lon
        gcp_rows = np.array([gcp.row for gcp in gcps])
        gcp_cols = np.array([gcp.col for gcp in gcps])
        
        # Sort GCPs by row and col for RectBivariateSpline (requires sorted unique grid)
        unique_rows = np.sort(np.unique(gcp_rows))
        unique_cols = np.sort(np.unique(gcp_cols))
        
        # Reshape GCP lat/lon into 2D grids (n_rows x n_cols)
        lat_grid_gcp = np.zeros((len(unique_rows), len(unique_cols)))
        lon_grid_gcp = np.zeros((len(unique_rows), len(unique_cols)))
        
        for gcp in gcps:
            row_idx = np.argmin(np.abs(unique_rows - gcp.row))
            col_idx = np.argmin(np.abs(unique_cols - gcp.col))
            lat_grid_gcp[row_idx, col_idx] = gcp.y
            lon_grid_gcp[row_idx, col_idx] = gcp.x
        
        # Create spline interpolators
        lat_spline = RectBivariateSpline(unique_rows, unique_cols, lat_grid_gcp, kx=3, ky=3)
        lon_spline = RectBivariateSpline(unique_rows, unique_cols, lon_grid_gcp, kx=3, ky=3)
        
        # Interpolate lat/lon for all pixel centers
        pixel_rows = np.arange(da.y.size) - 0.5
        pixel_cols = np.arange(da.x.size) - 0.5
        
        lat_grid = lat_spline(pixel_rows, pixel_cols)
        lon_grid = lon_spline(pixel_rows, pixel_cols)
        
        interpolated_lat = xr.DataArray(lat_grid.astype(np.float32), dims=["y", "x"])
        interpolated_lon = xr.DataArray(lon_grid.astype(np.float32), dims=["y", "x"])
    else:
        # GEOSAT-2: Use affine transform + CRS transformation (UTM → WGS84)
        # Generate pixel center coordinates (row, col) as meshgrid
        cols = np.arange(da.x.size)
        rows = np.arange(da.y.size)
        col_grid, row_grid = np.meshgrid(cols, rows)
        
        # Transform pixel centers to projected coordinates using affine transform
        # offset='center' gives pixel center coordinates
        easting, northing = rasterio.transform.xy(src_transform, row_grid, col_grid, offset='center')
        
        # Transform from projected CRS (UTM) to WGS84 (lat/lon)
        from rasterio.warp import transform
        lons, lats = transform(src_crs, 'EPSG:4326', easting.ravel(), northing.ravel())
        lons = np.array(lons).reshape(row_grid.shape)
        lats = np.array(lats).reshape(row_grid.shape)
        
        interpolated_lat = xr.DataArray(lats.astype(np.float32), dims=["y", "x"])
        interpolated_lon = xr.DataArray(lons.astype(np.float32), dims=["y", "x"])

    da = da.rename({"y": "latitude", "x": "longitude"})

    gains = []
    offsets = []

    band_names = [band["description"] for band in attrs["spectral_bands"]]
    ltoa_unit = attrs["spectral_bands"][0]["unit"]
    assert ltoa_unit in [
        "W/m2/sr/m-6",
        "W m<sup>-2</sup> &micro;m<sup>-1</sup> str<sup>-1</sup>",
    ]
    ltoa_unit = "W/m2/sr/µm"

    for band in attrs["spectral_bands"]:
        offsets.append(float(band["physical_bias"]))
        gains.append(float(band["physical_gain"]))


    offsets = xr.DataArray(offsets, dims=["bands"])
    gains = xr.DataArray(gains, dims=["bands"])
    shape = (da.latitude.size, da.longitude.size)
    dims = ["y", "x"]

    # incidence_angle = vza
    # https://docs.up42.com/data/viewing-angles
    sza_val = 90.0 - float(attrs["acquisition"]["SUN_ELEVATION"])
    saa_val = float(attrs["acquisition"]["SUN_AZIMUTH"])
    vza_val = float(attrs["acquisition"]["incidence_angle"])

    # Check that view zenith angle is small
    assert vza_val < 20.
    # Azimuth angle is not provided : set it to zero
    vaa_val = float(0.)

    sza = xr.DataArray(np.full(shape, sza_val), dims=dims, name="sza")

    saa = xr.DataArray(np.full(shape, saa_val), dims=dims, name="saa")

    vza = xr.DataArray(np.full(shape, vza_val), dims=dims, name="vza")
    
    vaa = xr.DataArray(np.full(shape, vaa_val), dims=dims, name="vaa")

    # DIMAP radiometric calibration differs between sensors:
    # GEOSAT-1 (DE-1): DIMAP V1 format, gain > 1  ->  LTOA = DN / gain + bias
    #   Per DIMAP V1 spec: "Formulae L=DN/GAIN+BIAS" (see eoreader spot45_product.py / dimap_v2_product.py)
    # GEOSAT-2 (DE-2): gain << 1, bias ~ -gain  ->  LTOA = DN * gain + bias
    #   (see eoreader gs2_product.py: bias + band_arr * gain)
    if gains.values[0] > 1:
        # GEOSAT-1 convention (DIMAP V1)
        da2 = da / gains + offsets
    else:
        # GEOSAT-2 convention
        da2 = da * gains + offsets

    cwav_map = {"NIR": 830, "Red": 660, "RED": 660, "Green": 560, "GREEN": 560, "Blue": 490, "BLUE": 490}
    cwav = np.array([cwav_map[x] for x in band_names])

    final_ds = xr.Dataset(
        data_vars={
            str(n.ltoa): (("bands", "y", "x"), da2.data.astype(np.float32)),
            str(n.sza): sza.astype(n.sza.dtype),
            str(n.saa): saa.astype(n.saa.dtype),
            str(n.vza): vza.astype(n.vza.dtype),
            str(n.vaa): vaa.astype(n.vaa.dtype),
            str(n.cwav): (("bands"), cwav.astype(np.int32)),
            str(n.lat): interpolated_lat.astype(n.lat.dtype),
            str(n.lon): interpolated_lon.astype(n.lon.dtype),
        },
        coords={
            "y": np.arange(shape[0]),
            "x": np.arange(shape[1]),
            "bands": band_names,
        },
        attrs=attrs["identification"] | attrs["geospatial"] |attrs["acquisition"] | {"spectral_bands": attrs["spectral_bands"]},
    ).chunk(chunks)

    final_ds[str(n.ltoa)].attrs["units"] = ltoa_unit

    for v in [n.rtoa, n.lon, n.lat, n.sza, n.saa, n.vza, n.cwav, n.wav]:
        if str(v) in final_ds:
            final_ds[str(v)].attrs = v.attrs
            
    imaging_date = final_ds.attrs['imaging_date']
    imaging_time = final_ds.attrs.get('imaging_time')
    if imaging_time:
        combined_str = f"{imaging_date} {imaging_time}"
    else:
        combined_str = imaging_date

    final_ds.attrs[n.datetime] = combined_str
    del final_ds.attrs["imaging_date"], final_ds.attrs["imaging_time"]

    if meta_data or subset_meta_data != []:
        dict_ = parse(open(file_dim).read())
        
        final_ds.attrs = final_ds.attrs | iter_through_subset(dict_, subset_meta_data)
    
    final_ds.attrs.update({"_srf_getter": "eoread.geosat.read_geosat_srf"})
    # Determine sensor type from number of spectral bands
    # GEOSAT-1 (DE-1): 3 bands, GEOSAT-2 (DE-2): 4 bands
    num_bands = len(attrs["spectral_bands"])
    sensor_id = "DE-2" if num_bands >= 4 else "DE-1"
    final_ds.attrs.update({"_srf_getter_arg": sensor_id})

    # Reverse the "bands" dimension
    final_ds = final_ds.isel(bands=slice(None, None, -1))
    
    return final_ds


def get_sample(kind: int) -> Path:
    """
    Download GEOSAT-1 or GEOSAT-2 sample products

    Parameters
    ----------
    kind : int
        1 or 2, indicating GEOSAT-1 or GEOSAT-2.
    """

    if kind == 1:
        url = "https://earth.esa.int/eogateway/ftp/missions/sample-data/third-party-missions/geosat/geosat-1/DE01_SL6_22S_1R_20220829T063018_20220829T063109_DMI_0_bb31.zip"
    elif kind == 2:
        url = "https://earth.esa.int/eogateway/ftp/missions/sample-data/third-party-missions/geosat/geosat-2/DE2_MS4_L1C_000000_20230523T092838_20230523T092841_DE2_48370_57E5.zip"
    else:
        # https://earth.esa.int/eogateway/missions/geosat-2/sample-data
        raise NotImplementedError

    return uncompress_decorator()(download_url)(
        url, getdir("DIR_SAMPLES") / f"GEOSAT-{kind}"
    )


def read_geosat_srf(
    sensor: Literal['DE-1', 'DE-2'],
    file_path: str | Path | None = None,
    panchromatic: bool = False,
) -> xr.Dataset:
    """
    Read GEOSAT-2 Spectral Response Function (SRF) data from Excel file.
    
    Parameters
    ----------
    sensor : {'DE-1', 'DE-2'}
        Sensor identifier.
    file_path : Path or str, optional
        Path to the Excel file containing SRF data. Defaults to
        'data/srfs/SpectralResponses_GEOSAT-2.xlsx'.
    panchromatic : bool, optional
        Whether to return panchromatic SRF data.
    
    Returns
    -------
    xr.Dataset
        Dataset containing SRF data with:
            - One variable for each band containing SRF response values
            - Each band has its own 'wavelengths' coordinate array
    """
    if file_path is None:
        url = "https://github.com/hygeos/eoread/releases/download/root/SpectralResponses_GEOSAT-2.xlsx"
        file_path = download_url(url, dirname=getdir("DIR_STATIC") / "srf" / "geosat")

    # first sheet is de2 (GEOSAT-2)
    if sensor == "DE-1":
        sheet = 1
        num_bands = 3
    elif sensor == "DE-2":
        sheet = 0
        num_bands = 5
    else:
        raise ValueError
        
    df = pd.read_excel(file_path, sheet_name=sheet, header=None)
    
    # Extract band names from first row, assuming they are in odd columns starting from index 1
    band_names = df.iloc[0, 1::2].values
    band_names = [name for name in band_names if not (pd.isna(name) or name == '')][:num_bands]
    
    # Normalize band names to match DIMAP band descriptions (case-sensitive)
    # GEOSAT-1 SRF uses "D1 NIR", "D1 RED", "D1 GREEN" -> DIMAP: "NIR", "Red", "Green"
    BAND_NAME_MAP = {
        "D1 NIR": "NIR",
        "D1 RED": "Red",
        "D1 GREEN": "Green",
    }

    # Extract SRF data for each band and create DataArrays
    data_vars = {}
    coords = {}
    band_central_wavelengths = {}
    for i, band_name in enumerate(band_names):

        if not panchromatic and band_name.upper() == "PAN":
            continue

        # Normalize band name
        band_name = band_name.strip()
        band_name = BAND_NAME_MAP.get(band_name, band_name)

        # Wavelengths for this band (even columns: 0, 2, 4, ...)
        wav_raw = pd.to_numeric(df.iloc[1:, 2*i], errors="coerce").values

        # SRF data for this band (odd columns: 1, 3, 5, ...)
        srf_raw = pd.to_numeric(df.iloc[1:, 2*i+1], errors="coerce").values

        # Filter out rows where either wavelength or SRF is NaN (handles varying-length bands)
        valid = ~np.isnan(wav_raw) & ~np.isnan(srf_raw)
        wavelengths = wav_raw[valid]
        band_data = srf_raw[valid].tolist()

        coord_name = f'wav_{band_name}'
        coords[coord_name] = wavelengths
        data_vars[band_name] = ([coord_name], band_data)

        assert len(band_data) == len(wavelengths)

        # Compute SRF-weighted average wavelength for this band
        srf = np.array(band_data, dtype=float)
        srf_sum = srf.sum()
        if srf_sum > 0:
            band_central_wavelengths[band_name] = float((wavelengths * srf).sum() / srf_sum)
        else:
            band_central_wavelengths[band_name] = float(wavelengths.mean())

    # Sort band names by increasing weighted-average wavelength
    sorted_band_names = sorted(data_vars.keys(), key=lambda b: band_central_wavelengths[b])

    # Create Dataset with dimensions for each band's wavelengths
    ds = xr.Dataset(data_vars, coords=coords)

    # Set wavelength attributes
    for coord in ds.coords:
        ds[coord].attrs.update(units='nm')

    return ds[sorted_band_names]
