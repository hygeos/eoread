from pathlib import Path

import numpy as np
import xarray as xr

from core.geo.naming import names
from core.tools import drop_unused_dims
from core import env, log

from eoread.flags import FlagsReaderBase
from eoread.tools import filter_metadata, format_chunks


# Nominal central wavelengths (nm) of the HICO bands, in product order.
# These identifiers are shared with NASA's iss_hico RSR table
# (https://oceandata.sci.gsfc.nasa.gov/images/rsr/iss_hico_RSR.nc) and with
# polymer's `bands_hico` list.
HICO_BANDS = [
    353, 358, 364, 370, 375, 381, 387, 393, 398, 404, 410,
    416, 421, 427, 433, 438, 444, 450, 456, 461, 467, 473,
    479, 484, 490, 496, 501, 507, 513, 519, 524, 530, 536,
    542, 547, 553, 559, 564, 570, 576, 582, 587, 593, 599,
    605, 610, 616, 622, 627, 633, 639, 645, 650, 656, 662,
    668, 673, 679, 685, 690, 696, 702, 708, 713, 719, 725,
    731, 736, 742, 748, 753, 759, 765, 771, 776, 782, 788,
    794, 799, 805, 811, 816, 822, 828, 834, 839, 845, 851,
    857, 862, 868, 874, 880, 885, 891, 897, 902, 908, 914,
    920, 925, 931, 937, 943, 948, 954, 960, 965, 971, 977,
    983, 988, 994, 1000, 1006, 1011, 1017, 1023, 1028, 1034, 1040,
    1046, 1051, 1057, 1063, 1069, 1074, 1080,
]


def Level1_HICO(
        filepath: str | Path,
        chunks: int | tuple = 500,
        metadata_template: list = None,
        verbose: bool = True,
    ) -> xr.Dataset:
    """
    Read an ISS/HICO Level-1B product as an xarray.Dataset.

    HICO (Hyperspectral Imager for the Coastal Ocean) is a hyperspectral
    imaging spectrometer that flew on the International Space Station
    (2011-2018). The Level-1B products distributed by OB.DAAC are netCDF
    files containing top of atmosphere radiance in 128 bands (353-1080 nm).

    The dataset contains TOA radiances, viewing/solar angles on the full
    grid, and geolocation information.

    Parameters
    ----------
    filepath : Path or str
        Path to the HICO Level-1B netCDF file (H*.L1B_ISS.nc).
    chunks : int or tuple, optional
        Size of chunks for spatial dimensions. If int, applies to both
        dimensions. Default is 500.
    metadata_template : list, optional
        Template to filter the product metadata stored in the dataset
        attributes. If None (default), all metadata is kept.

    Returns
    -------
    xr.Dataset
        Dataset with variables Ltoa, latitude, longitude, sza, vza, saa,
        vaa, cwav and altitude, on dimensions (y, x, bands).

    Raises
    ------
    AssertionError
        If the file does not exist or is not a HICO Level-1B product.

    Examples
    --------
    >>> ds = Level1_HICO('H2014069115230.L1B_ISS.nc', chunks=1000)
    """
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exist'

    # Format chunks
    chunks = format_chunks(chunks)

    if verbose: log.debug('Open HICO product')
    ds_root = xr.open_datatree(filepath)
    ds_products = ds_root["products"].to_dataset()
    ds_nav = ds_root["navigation"].to_dataset()
    ds_time = (ds_root["metadata"]["FGDC"]["Identification_Information"]
               ["Time_Period_of_Content"])

    # Check that this is a HICO product with the expected number of bands
    lt = ds_products["Lt"]
    assert str(lt.attrs.get("units")) == "W/m^2/micrometer/sr", \
        f'Unexpected units for Lt: {lt.attrs.get("units")}'
    wavelengths = np.asarray(lt.attrs["wavelengths"], dtype="float32")
    assert len(wavelengths) == len(HICO_BANDS), \
        f'Expected {len(HICO_BANDS)} bands, got {len(wavelengths)}'

    ds = xr.Dataset()

    # Geolocation and angles (in standard geographic order, no flip needed)
    if verbose: log.debug('Read geolocation and angles')
    ds[str(names.lat)] = ds_nav["latitudes"].astype("float32")
    ds[str(names.lon)] = ds_nav["longitudes"].astype("float32")
    ds[str(names.sza)] = ds_nav["solar_zenith"].astype("float32")
    ds[str(names.vza)] = ds_nav["sensor_zenith"].astype("float32")
    ds[str(names.saa)] = ds_nav["solar_azimuth"].astype("float32")
    ds[str(names.vaa)] = ds_nav["sensor_azimuth"].astype("float32")

    # Top of atmosphere radiance
    if verbose: log.debug('Read top of atmosphere radiance')
    ds[str(names.ltoa)] = lt
    ds = ds.rename(scan_lines=str(names.rows), samples=str(names.columns))

    # Band identifiers: nominal central wavelengths (as strings), in product
    # order. The actual central wavelengths are stored in `cwav`.
    ds = ds.assign_coords({str(names.bands): [str(b) for b in HICO_BANDS]})
    ds[str(names.ltoa)].attrs["units"] = "W/m^2/micrometer/sr"

    # Central (nominal) wavelengths, from the product
    if verbose: log.debug('Extract central wavelengths')
    ds = ds.assign({
        str(names.cwav): ((str(names.bands),), wavelengths,
                          {"units": "nm", "desc": "Central wavelength"}),
    })

    # Altitude: HICO products carry no terrain information (polymer v4 also
    # processed HICO with a zero altitude)
    ds["altitude"] = xr.zeros_like(ds[str(names.lat)], dtype="float32")
    ds["altitude"].attrs.update({"units": "m", "desc": "Terrain altitude"})

    # Chunking
    ds = ds.chunk(chunks).chunk({str(names.bands): -1})

    # Attributes
    if verbose: log.debug('Add important attributes')
    datetime_str = (f"{ds_time.attrs['Beginning_Date']}"
                    f"{ds_time.attrs['Beginning_Time']}")
    from datetime import datetime as _datetime
    ds.attrs[str(names.datetime)] = \
        _datetime.strptime(datetime_str, "%Y%m%d%H%M%S").isoformat()
    ds.attrs[str(names.sensor)] = "HICO"
    ds.attrs[str(names.platform)] = "ISS"
    ds.attrs[str(names.resolution)] = 30
    ds.attrs[str(names.product_name)] = filepath.name
    ds.attrs[str(names.input_directory)] = str(filepath.parent)

    # Spectral response functions from NASA OB.DAAC (iss_hico_RSR.nc)
    ds.attrs["_srf_getter"] = "eotools.srf.get_SRF_NASA"
    ds.attrs["_srf_getter_arg"] = "iss_hico"

    # Flags: HICO Level-1B products carry no per-pixel quality flags
    ds.attrs["_flag_reader"] = "eoread.hico.FlagsReader_HICO"

    filter_fn = (lambda x, y: x) if metadata_template is None else filter_metadata
    ds.attrs["metadata"] = filter_fn(dict(ds_root.attrs), metadata_template)

    # All bands belong to a single group. Assigned after drop_unused_dims,
    # which removes non-dimension coordinates (core convention).
    ds = drop_unused_dims(ds)
    ds = ds.assign_coords({
        str(names.bgroup): (str(names.bands), ["bands"] * len(HICO_BANDS)),
    })

    # eoread convention: 3D rasters in bands-first order
    ds = ds.transpose(..., str(names.rows), str(names.columns))

    return ds.unify_chunks()


def get_sample(level: int = 1) -> Path:
    """
    Retrieve a sample HICO Level-1B product file for testing.

    The path is read from the environment variable LEVEL1_HICO (see the
    `.env` file of this repository).

    Returns
    -------
    Path
        Path to the HICO Level-1B netCDF file.

    Raises
    ------
    AssertionError
        If the sample product does not exist.
    """
    assert level == 1, 'Only Level-1 products are supported'
    sample = Path(env.getvar("LEVEL1_HICO"))
    assert sample.exists(), f'Sample product does not exist: {sample}'
    return sample


class FlagsReader_HICO(FlagsReaderBase):
    """
    Flags reader for ISS/HICO Level-1B data.

    HICO Level-1B products do not contain per-pixel quality flags (quality
    control is applied by the Level-2 processor), so all standard flags are
    returned as False. This matches the behavior of polymer v4, which
    initialized HICO bitmasks to zero and applied a separate land mask.
    """

    def requires(self) -> list[str]:
        """List of variables in Dataset required by `getflag`."""
        return []

    def dims_like(self) -> str:
        """Returns an input var name that has the same shape as the output"""
        # Per-pixel flags are 2D (a 3D reference such as Ltoa would give
        # per-band flags, which is not what downstream processors expect)
        return str(names.lat)

    def getflag(self, ds: xr.Dataset, flag_name) -> xr.DataArray:
        """
        Retrieve a flag from the dataset using the standard flag name.

        Always returns an all-False mask (no quality flags in the product).
        """
        return xr.zeros_like(ds[self.dims_like()], dtype="bool")
