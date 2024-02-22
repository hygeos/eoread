import warnings
from typing import Any, Dict, List, Literal, Optional
from packaging import version

import numpy as np
import pandas as pd
import xarray as xr


def interp(
    da: xr.DataArray,
    *,
    sel: Optional[Dict[str, xr.DataArray]] = None,
    interp: Optional[Dict[str, xr.DataArray]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    """Interpolate or select a DataArray onto new coordinates

    This function is similar to xr.interp and xr.sel, but:
        - Supports dask-based inputs (in sel and interp) without
          triggering immediate computation
        - Supports both selection and indexing
        - Does not use xarray's default .interp method (improved efficiency)

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray
    sel : Optional[Dict[str, xr.DataArray]], optional
        A dict mapping dimensions names to values, by default None
    interp : Optional[Dict[str, xr.DataArray]], optional
        A dict mapping dimension names to new coordinates, by default None
    options : Optional[Dict[str, Any]], optional
        A dict mapping dimension names to a dictionary of options to use for
        the current sel or interp.
        For sel dimensions, the options are passed to `pandas.Index.get_indexer`. In
        particular, method={None [default, raises an error if values not in index],
        "pad"/"ffill", "backfill"/"bfill", "nearest"}
        For interp dimensions:
            `bounds`: behaviour in case of out-of-bounds values (default "error")
                - "error": raise an error
                - "nan": set NaN values
                - "clip": clip values within the bounds
            `skipna`: whether to skip input NaN values (default True)

    Example
    -------
    >>> index(
    ...     data,  # input DataArray with dimensions (a, b, c)
    ...     interp={ # interpolation dimensions
    ...         'a': a_values, # `a_values` is a DataArray with dimension (x, y)
    ...     },
    ...     sel={ # selection dimensions
    ...         'b': b_values, # `b_values` is a DataArray with dimensions (x)
    ...     },
    ...     options={ # define options per-dimension
    ...         'a': {"bounds": "clip"},
    ...         'b': {"method": "nearest"},
    ...     },
    ... ) # returns a DataArray with dimensions (x, y, c)
    No interpolation or selection is performed along dimension `c` thus it is
    left as-is.

    Returns
    -------
    xr.DataArray
        New DataArray on the new coordinates.
    """
    assert version.parse(xr.__version__) >= version.parse("2024.01.0")

    assert (da.chunks is None) or (
        len(da.chunks) == 0
    ), "Input DataArray should not be dask-based"

    sel = sel or {}
    interp = interp or {}

    # group all sel+interp dimensions
    ds = xr.Dataset({**sel, **interp})

    # prevent common dimensions between da and sel+interp
    assert not set(ds.dims).intersection(da.dims)

    # transpose them to ds.dims
    ds = ds.transpose(*ds.dims)

    ret = xr.map_blocks(
        index_block,
        ds,
        kwargs={
            "data": da,
            "dims_sel": sel.keys(),
            "dims_interp": interp.keys(),
            "options": options,
        },
    )
    ret.attrs.update(da.attrs)

    return ret


def broadcast_numpy(ds: xr.Dataset, dims) -> Dict:
    """
    Returns all data variables in `ds` as numpy arrays
    broadcastable against the dimensions defined by dims
    (with new single-element dimensions)

    This requires the input to be broadcasted to common dimensions.
    """
    result = {}
    for var in ds:
        result[var] = ds[var].data[
            tuple([slice(None) if d in ds[var].dims else None for d in dims])
        ]
    return result


def broadcast_shapes(ds: xr.Dataset, dims) -> Dict:
    """
    For each data variable in `ds`, returns the shape for broadcasting
    in the dimensions defined by dims
    """
    result = {}
    for var in ds:
        result[var] = tuple(
            [
                ds[var].shape[ds[var].dims.index(d)] if d in ds[var].dims else 1
                for d in dims
            ]
        )
    return result


def index_block(
    ds: xr.Dataset,
    data: xr.DataArray,
    dims_sel: List,
    dims_interp: List,
    options: Optional[Dict] = None,
) -> xr.DataArray:
    """
    This function is called by map_blocks in function `index`, and performs the
    indexing and interpolation at the numpy level.
    """
    dims_sel_interp = list(dims_sel) + list(dims_interp)
    options = options or {}

    # determine output dimensions based on numpy's advanced indexing rules
    out_dims = []
    out_shape = []
    dims_added = False
    for dim in data.dims:
        if dim in dims_sel_interp:
            if not dims_added:
                out_dims.extend(list(ds.dims))
                out_shape.extend(list(ds.dims.values()))
                dims_added = True
        else:
            out_dims.append(dim)
            out_shape.append(data[dim].size)

    # get broadcasted data from ds (all with the same number of dimensions)
    np_indexers = broadcast_numpy(ds, ds.dims)
    x_indexers = broadcast_shapes(ds, out_dims)

    keys = [slice(None)] * data.ndim

    # selection keys (non-interpolation dimensions)
    for dim in dims_sel:
        idim = data.dims.index(dim)

        # default sel options
        opt = {**(options[dim] if dim in options else {})}
        keys[idim] = (
            data.indexes[dim]
            .get_indexer(np_indexers[dim].ravel(), **opt)
            .reshape(np_indexers[dim].shape)
        )
        if ((keys[idim]) < 0).any():  # type: ignore
            raise ValueError(
                f"Error in selection of dimension {dim} with options={opt}"
            )

    # determine bracketing values and interpolation ratio
    # for each interpolation dimension
    iinf = {}
    x_interp = {}
    for dim in dims_interp:
        # default interp options
        opt = {
            "bounds": "error",
            "skipna": True,
            **(options[dim] if dim in options else {}),
        }
        assert opt["bounds"] in ["error", "nan", "clip"]

        iinf[dim] = (
            data.indexes[dim]
            .get_indexer(np_indexers[dim].ravel(), method="ffill")
            .reshape(np_indexers[dim].shape)
        )

        # Clip indices
        iinf[dim] = iinf[dim].clip(0, len(data.indexes[dim]) - 2)

        iinf_dims_out = iinf[dim].reshape(x_indexers[dim])
        vinf = data.indexes[dim].values[iinf_dims_out]
        vsup = data.indexes[dim].values[iinf_dims_out + 1]
        x_interp[dim] = np.array(
            np.clip(
                (np_indexers[dim].reshape(x_indexers[dim]) - vinf) / (vsup - vinf), 0, 1
            )
        )

        # skip nan values
        if opt["skipna"]:
            isnan = np.isnan(np_indexers[dim])
        else:
            isnan = np.array(False)

        # deal with out-of-bounds (non-nan) values
        if opt["bounds"] in ["error", "nan"]:
            valid = in_index(data.indexes[dim], np_indexers[dim])

            if opt["bounds"] == "error":
                if (~valid & ~isnan).any():
                    vmin = np_indexers[dim][~isnan].min()
                    vmax = np_indexers[dim][~isnan].max()
                    raise ValueError(
                        f"out of bounds values [{vmin} -> {vmax}] during interpolation "
                        f"of {data.name} in dimension {dim} "
                        f"[{data.indexes[dim][0]}, {data.indexes[dim][-1]}] "
                        f"with options={opt}"
                    )

            # opt['bounds'] == "nan"
            x_interp[dim][(~valid | isnan).reshape(x_indexers[dim])] = np.NaN

        elif opt["skipna"]:
            x_interp[dim][isnan.reshape(x_indexers[dim])] = np.NaN

    # loop over the 2^n bracketing elements
    # (cartesian product of [0, 1] over n dimensions)
    result = 0
    data_values = data.values
    for b in range(2 ** len(dims_interp)):

        # dim loop
        coef = 1
        for i, dim in enumerate(dims_interp):

            # bb is the ith bit in b (0 or 1)
            bb = ((1 << i) & b) >> i
            x = x_interp[dim]
            if bb:
                # TODO: can we use += here ?
                coef = coef * x
            else:
                coef = coef * (1 - x)

            keys[data.dims.index(dim)] = iinf[dim] + bb

        result += coef * data_values[tuple(keys)]

    # determine output coords
    coords = {}
    for dim in out_dims:
        if dim in data.coords:
            coords[dim] = data.coords[dim]
        elif dim in ds.coords:
            coords[dim] = ds.coords[dim]

    ret = xr.DataArray(
        result,
        dims=out_dims,
        coords=coords,
    )

    return ret


def index(*args, **kwargs):
    warnings.warn("This function has been renamed to `interp`", DeprecationWarning)
    return interp(*args, **kwargs)


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
    warnings.warn(
        "Deprecated function: use function `interp` instead", DeprecationWarning
    )

    # Check that aux is not dask-based
    assert (da.chunks is None) or (
        len(da.chunks) == 0
    ), "Input DataArray should not be dask-based"

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
        xr.Dataset(kwargs),  # use dim names from source Dataarray
        template=template,
        kwargs={"aux": da},
    )

    return interpolated


def interp_chunk(ds, aux):
    """
    Apply da.interp to a single chunk
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return aux.interp({k: ds[k] for k in ds}).reset_coords(drop=True)


def sel_chunk(ds, aux):
    """
    da.sel of a single chunk
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return aux.sel({k: ds[k] for k in ds}, method="nearest").reset_coords(drop=True)


def interp_legacy(
    aux: xr.DataArray,
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
        "Deprecated function: use function `interp` instead", DeprecationWarning
    )

    def interp_chunk(ds):
        """
        Interpolation of a single chunk
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return aux.interp({k: ds[v] for (k, v) in dims.items()}).reset_coords(
                drop=True
            )

    # Check that aux is not dask-based
    assert (aux.chunks is None) or (
        len(aux.chunks) == 0
    ), "Auxiliary DataArray should not be dask-based"

    first_dim = list(dims.values())[0]
    template = template or ds_coords[first_dim].reset_coords(drop=True)

    # update the interpolated with input attributes
    template.attrs = aux.attrs

    interpolated = xr.map_blocks(
        interp_chunk, ds_coords[dims.keys()], template=template
    )

    return interpolated


def in_index(ind: pd.Index, values: np.ndarray):
    """
    Returns whether each value is within the range defined by `ind`
    """
    if ind.is_monotonic_increasing:
        vmin = ind[0]
        vmax = ind[-1]

    elif ind.is_monotonic_decreasing:
        vmin = ind[-1]
        vmax = ind[0]

    else:
        raise ValueError(
            f"Index of dimension {ind.name} should be either "
            "monotonically increasing or decreasing."
        )

    return (values >= vmin) & (values <= vmax)
