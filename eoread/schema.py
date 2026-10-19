#!/usr/bin/env python3
"""
Validation of the canonical output schema.

Every v2 reader must emit a dataset conforming to the locked canonical schema
(see REFACTORING.md, §5). Canonical variable names and dtypes come from
`core.geo.naming.names`, the single source of truth; this module only *checks*
and never rewrites data.

`validate()` collects every violation and raises a single `SchemaError` listing
them all, so a reader port fails loudly with the full picture at once.
"""

from datetime import datetime

import numpy as np
import xarray as xr

from core.geo.naming import names


#: Version stamped on datasets by `Reader.finalize()`
SCHEMA_VERSION = '2.0'


class SchemaError(ValueError):
    """Raised when a dataset violates the canonical output schema."""


#: The payload variables; exactly one is the declared primary per reader class.
#: 'LST' is a single-layer (y, x) product field (ECOSTRESS LSTE) that carries
#: no bands dimension; `lst` to be proposed upstream in core.geo.naming.names.
PAYLOADS = (str(names.ltoa), str(names.rtoa), str(names.bt), 'LST', str(names.rho_w),
            'rho_surface', 'rho_flat')

#: 2D geometry/angle variables: (registry name, mandatory)
GEOMETRY_VARS = (
    (names.lat, True),
    (names.lon, True),
    (names.sza, False),
    (names.saa, False),
    (names.vza, False),
    (names.vaa, False),
)

#: Variables that must never appear in a v2 dataset: flags are decoded on
#: demand from the `_flag_reader` attribute (see eoread.flags)
FORBIDDEN_VARS = (str(names.flags), str(names.quality))

#: Mandatory dataset attributes and their expected type(s)
REQUIRED_ATTRS = {
    str(names.sensor): str,
    str(names.platform): str,
    str(names.product_name): str,
    str(names.datetime): str,
    str(names.input_directory): str,
    str(names.resolution): (int, float),
    str(names.shortname): str,
    str(names.crs): str,
}


def _is_string_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)


def validate(ds: xr.Dataset, primary=None, required_vars=()) -> None:
    """
    Validate a v2 reader output against the canonical schema.

    Checks presence/dims/dtype of variables and the mandatory attribute set
    (never data values). All violations are collected and reported together.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check, typically `Reader.ds` after `finalize()`.
    primary : str or None
        Declared primary payload variable ('Ltoa', 'Rtoa', 'BT' or 'LST').
        When None, exactly one of PAYLOADS must be present.
    required_vars : sequence of str
        Variables this reader must emit (presence checked only).

    Raises
    ------
    SchemaError
        Listing every violation found.
    """
    errors = []

    # --- dimensions -------------------------------------------------------
    for dim in (names.rows, names.columns):
        if str(dim) not in ds.dims:
            errors.append(f'missing dimension {dim!s}')

    # --- geometry and angles ----------------------------------------------
    grid_dims = (str(names.rows), str(names.columns))
    for name, mandatory in GEOMETRY_VARS:
        if str(name) not in ds:
            if mandatory:
                errors.append(f'missing required variable {name!s}')
            continue
        var = ds[str(name)]
        if tuple(var.dims) != grid_dims:
            errors.append(f'{name}: dims are {tuple(var.dims)}, expected {grid_dims}')
        if name.dtype and var.dtype != np.dtype(name.dtype):
            errors.append(f'{name}: dtype is {var.dtype}, expected {name.dtype}')

    # --- payload ------------------------------------------------------------
    present = [p for p in PAYLOADS if p in ds]
    if primary is None:
        if len(present) != 1:
            errors.append(f'expected exactly one of {PAYLOADS} as payload, got {present}')
    else:
        primary = str(primary)
        if primary not in PAYLOADS:
            raise ValueError(f"invalid primary payload {primary!r}, must be one of {PAYLOADS}")
        if primary not in present:
            errors.append(f'missing declared primary payload {primary}')
    for p in present:
        var = ds[p]
        # Single-layer payloads (e.g. LST) carry no bands dimension.
        expected_dims = ((str(names.bands),) if str(names.bands) in var.dims
                         else ()) + grid_dims
        if tuple(var.dims) != expected_dims:
            errors.append(f'{p}: dims are {tuple(var.dims)}, expected {expected_dims}')
        if str(names.unit) not in var.attrs:
            errors.append(f'{p}: missing {names.unit!s} attribute')

    # --- bands --------------------------------------------------------------
    if str(names.bands) in ds.dims or str(names.bands) in ds:
        if str(names.bands) not in ds:
            errors.append(f'missing coordinate {names.bands!s}: band labels are mandatory')
        else:
            bcoord = ds[str(names.bands)]
            if not _is_string_dtype(bcoord.dtype):
                errors.append(f'{names.bands}: coordinate should carry band names (string dtype), got {bcoord.dtype}')
            elif len(set(bcoord.values)) != bcoord.size:
                errors.append(f'{names.bands}: band labels are not unique')
            if str(names.cwav) not in ds:
                errors.append(f'missing required variable {names.cwav!s}')
            else:
                cw = ds[str(names.cwav)]
                if tuple(cw.dims) != (str(names.bands),):
                    errors.append(f'{names.cwav}: dims are {tuple(cw.dims)}, expected ({names.bands},)')
                elif not np.issubdtype(cw.dtype, np.floating):
                    errors.append(f'{names.cwav}: dtype is {cw.dtype}, expected a floating point type')
                else:
                    vals = cw.values
                    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
                        errors.append(f'{names.cwav}: values must be finite and strictly positive')
            if str(names.bgroup) in ds:
                bg = ds[str(names.bgroup)]
                if tuple(bg.dims) != (str(names.bands),):
                    errors.append(f'{names.bgroup}: dims are {tuple(bg.dims)}, expected ({names.bands},)')
                elif not _is_string_dtype(bg.dtype):
                    errors.append(f'{names.bgroup}: should carry band group names (string dtype), got {bg.dtype}')

    # --- forbidden variables --------------------------------------------------
    for name in FORBIDDEN_VARS:
        if name in ds.data_vars:
            errors.append(
                f'{name} must not be a data variable: flags are decoded on demand via the _flag_reader attribute'
            )

    # --- declared required variables ---------------------------------------------
    for name in required_vars:
        if str(name) not in ds:
            errors.append(f'missing required variable {str(name)}')

    # --- attributes -------------------------------------------------------
    attrs = ds.attrs
    for key, typ in REQUIRED_ATTRS.items():
        if key not in attrs:
            errors.append(f'missing dataset attribute {key!r}')
            continue
        value = attrs[key]
        if isinstance(value, bool) or not isinstance(value, typ):
            expected = typ.__name__ if isinstance(typ, type) else ' or '.join(t.__name__ for t in typ)
            errors.append(f'attribute {key!r} should be {expected}, got {type(value).__name__}')
    if attrs.get('schema_version') != SCHEMA_VERSION:
        errors.append(f'attribute "schema_version" is {attrs.get("schema_version")!r}, expected {SCHEMA_VERSION!r}')
    dt = attrs.get(str(names.datetime))
    if isinstance(dt, str):
        try:
            datetime.fromisoformat(dt)
        except ValueError:
            errors.append(f'attribute "datetime" should be ISO-8601, got {dt!r}')

    # --- report -----------------------------------------------------------
    if errors:
        raise SchemaError(
            f'dataset does not conform to the canonical schema (version {SCHEMA_VERSION}), '
            f'{len(errors)} violation(s):\n  - ' + '\n  - '.join(errors)
        )
