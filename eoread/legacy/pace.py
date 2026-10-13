from pathlib import Path
from typing import Union

import xarray as xr
from core.tools import only
from core.geo.naming import names
from dateutil.parser import parse

from eoread.tools import collect_sample, format_chunks, filter_metadata
from eoread.flags import GenericFlags, FlagsReaderBase


def Level1B_PACE_OCI(
        product_pace_oci: Path,
        chunks: int|list = 500,
        metadata_template: Union[list, None] = None,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read PACE OCI Level 1B products
    """
    
    # Open file
    tree = xr.open_datatree(product_pace_oci)
    
    # Format chunks
    chunks = format_chunks(chunks)

    geo = tree["geolocation_data"].to_dataset().reset_coords(["latitude", "longitude"])
    bdata = tree["sensor_band_parameters"].to_dataset()
    obs = tree["observation_data"].to_dataset()
    
    # Add latlon rasters, geometric angles and masks
    ds = xr.Dataset().assign(geo.astype('float32'))
    ds = ds.rename_vars({
        "latitude": str(names.lat),
        "longitude": str(names.lon),
        "sensor_azimuth": str(names.vaa),
        "sensor_zenith": str(names.vza),
        "solar_azimuth": str(names.saa),
        "solar_zenith": str(names.sza),
        "watermask": "water"
    })

    # TOA reflectance
    ds[str(names.rtoa)] = xr.concat(
        [
            _Internal.rename_bands(obs["rhot_blue"]),
            _Internal.rename_bands(obs["rhot_red"]),
            _Internal.rename_bands(obs["rhot_SWIR"]),
        ],
        dim=str(names.bands),
    )
    ds[str(names.rtoa)].attrs['unit'] = None
    
    # Determine central wavelengths
    ds[str(names.cwav)] = xr.concat(
        [
            _Internal.rename_bands(bdata["blue_wavelength"]),
            _Internal.rename_bands(bdata["red_wavelength"]),
            _Internal.rename_bands(bdata["SWIR_wavelength"]),
        ],
        dim=str(names.bands),
    )

    # Attributes
    ds.attrs[str(names.sensor)] = tree.attrs['instrument']
    ds.attrs[str(names.platform)] = tree.attrs['platform']
    ds.attrs[str(names.product_name)] = product_pace_oci.name
    ds.attrs[str(names.input_directory)] = str(product_pace_oci.parent)
    ds.attrs[str(names.resolution)] = 1200
    time_start = parse(tree.attrs["time_coverage_start"])
    time_end = parse(tree.attrs["time_coverage_end"])
    ds.attrs[str(names.datetime)] = (time_start + (time_end - time_start) / 2).isoformat()
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata'] = filter_fn(tree.attrs, metadata_template)
    ds.attrs['_flag_reader'] = 'eoread.pace.FlagsReader_PACE'

    # # SRF getter
    # ds.attrs['_srf_getter'] = 'None'
    # ds.attrs['_srf_getter_arg'] = ''
    
    # Add band names as coordinates
    ds = ds.assign_coords({str(names.bands): ds[str(names.cwav)].astype(int).astype(str)})
    ds = ds.chunk({str(names.bands): -1})

    # Rename and rechunk spatial dimensions
    ds = ds.rename(scans="y", pixels="x").chunk(chunks)

    # Remove duplicate bands by keeping only unique wavelength values, sorted by increasing wavelength
    seen = set()
    ds = ds.isel({str(names.bands): [
        i for i, w in sorted([
            (i, w) for i, w in enumerate(ds.bands.values)
            if w not in seen and not seen.add(w)
        ], key=lambda x: x[1])
    ]})

    return ds


def get_sample(level: int = 1) -> Path:
    """
    Return sample PACE Level-1B products.

    Parameters
    ----------
    level : int, optional
        Processing level of the product. Default is 1.

    Returns
    -------
    Path
        Path to the PACE product. Includes:
        - path: path to the product
        - roi: region of interest within the full product
        - px: sample pixel coordinates within the roi
    """
    return collect_sample(f'LEVEL{level}_PACE', 'nasa', 'PACE-OCI', level)


class FlagsReader_PACE(FlagsReaderBase):
    """
    Flags reader for PACE OCI (Ocean Color Instrument) data.
    
    Provides access to quality flags including land/water mask and
    quality indicators.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return ['quality_flag','water']  # Use viewing zenith angle as reference
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return 'quality_flag'
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the PACE dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            PACE dataset containing quality_flag and water variables.
        flag_name : GenericFlags
            Standard flag identifier (L1_INVALID or LAND).
        """
        if flag_name == GenericFlags.L1_INVALID:
            return ds['quality_flag']
        if flag_name == GenericFlags.LAND:
            return ~ds['water']
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")


################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def rename_bands(ds: xr.Dataset) -> xr.Dataset:
        """Rename band dimension and add band group coordinate."""
        
        # Determine band dimension
        dim = only([d for d in ds.dims if 'bands' in d])
        
        # Rename bands dimension
        ds = ds.rename({dim: str(names.bands)})
        
        # Assign bands group dimension
        bgroup = [dim] * len(ds[str(names.bands)])
        ds = ds.assign_coords({str(names.bgroup): (str(names.bands), bgroup)})
        
        return ds