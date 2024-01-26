from typing import Optional

import warnings
import xarray as xr


def interp(aux: xr.DataArray,
           ds_coords: xr.Dataset,
           dims: dict,
           template: Optional[xr.DataArray] = None,
           ) -> xr.DataArray:
    """
    Interpolation of xarray DataArray `aux` along dimensions provided
    in dask-based Dataset `ds_coords`. The mapping of these dimensions is
    defined in `dims`.

    The xarray `interp` method does not work efficiently on dask-based
    coordinates arrays (full arrays are computed); this function is
    designed to work around this.

    The mapping of variable names is provided in `dims`.

    `template`: the DataArray template for the output of this function.
    By default, use ds_coords[dims[0]].

    Example:
        interp(auxdata, ds, {'lat': 'latitude', 'lon': 'longitude'})

    This interpolates `auxdata` along dimensions 'lat' and 'lon', by chunks,
    with values defined in ds['latitude'] and ds['longitude'].
    """
    def interp_chunk(ds):
        """
        Interpolation of a single chunk
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return aux.interp(
                {k: ds[v]
                 for (k, v) in dims.items()
                 }).reset_coords(drop=True)

    # Check that aux is not dask-based
    assert (aux.chunks is None) or (len(aux.chunks) == 0), \
        'Auxiliary DataArray should not be dask-based'

    first_dim = list(dims.values())[0]
    template = template or ds_coords[first_dim].reset_coords(drop=True)

    # update the interpolated with input attributes
    template.attrs = aux.attrs

    interpolated = xr.map_blocks(
        interp_chunk,
        ds_coords[dims.keys()],
        template=template
    )

    return interpolated
