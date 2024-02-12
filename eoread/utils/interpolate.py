from typing import Literal, Optional

import warnings
import xarray as xr


def selinterp(
    da: xr.DataArray,
    *,
    method: Literal["interp", "nearest"],
    template: Optional[xr.DataArray] = None,
    **kwargs: xr.DataArray,
) -> xr.DataArray:
    """
    Interpolation (or selection) of xarray DataArray `da` along dimensions
    provided as kwargs.

    xarray's `interp` and `sel` methods do not work efficiently on dask-based
    coordinates arrays (full arrays are computed); this function is
    designed to work around this.

    `method`:
        "interp" to apply da.interp
        "nearest" to apply da.sel with method="nearest"

    `template`: the DataArray template for the output of this function.
    By default, use first kwarg.

    Example:
        selinterp(auxdata,
                  method='interp',
                  lat=ds.latitude,
                  lon=ds.longitude)

    This interpolates `auxdata` along dimensions 'lat' and 'lon', by chunks,
    with values defined in ds.latitude and ds.longitude.
    """

    # Check that aux is not dask-based
    assert (da.chunks is None) or (len(da.chunks) == 0), \
        'Input DataArray should not be dask-based'

    first_dim = list(kwargs.values())[0]
    template = template or first_dim.reset_coords(drop=True).rename(da.name).astype(
        {
            "interp": "float64",
            "nearest": da.dtype,
        }[method]
    )

    # update the template with input attributes
    template.attrs = da.attrs

    func = {
        "interp": interp_chunk,
        "nearest": sel_chunk,
    }[method]

    interpolated = xr.map_blocks(
        func,
        xr.Dataset(kwargs),
        template=template,
        kwargs={"aux": da, "dims": {k: v.name for k, v in kwargs.items()}},
    )

    return interpolated


def interp_chunk(ds, aux, dims):
    """
    Apply da.interp to a single chunk
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return aux.interp(
            {k: ds[v]
                for (k, v) in dims.items()
                }).reset_coords(drop=True)


def sel_chunk(ds, aux, dims):
    """
    da.sel of a single chunk
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return aux.sel(
            {k: ds[v] for (k, v) in dims.items()}, method="nearest"
        ).reset_coords(drop=True)


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
    warnings.warn(
        "Deprecated function: use function `selinterp` instead", DeprecationWarning
    )
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
