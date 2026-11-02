#!/usr/bin/env python3
"""
Validation of the canonical output schema.

Every v2 reader must emit a dataset conforming to the locked canonical schema
(see REFACTORING.md, §5). Canonical variable names and dtypes come from
`core.geo.naming.names`, the single source of truth; this module only *checks*
and never rewrites data.

Band rules checked here (see REFACTORING.md, §10): band labels are scientific
names (letter first; never wavelengths or bare numbers — the wavelength lives
in `cwav`), and the bands dimension is sorted by ascending central wavelength.

The single exception to the one-payload rule is the multi-resolution native
layout (MSI `resolution=None`): one payload per resolution group, named
`Ltoa_{res}m` / `Rtoa_{res}m`, each living on its own `(bands_{res}m, y_{res}m,
x_{res}m)` grid with a per-group `cwav_{res}m` table; the finest group is the
declared primary payload. The same band rules apply to each group's band dim.

`validate()` collects every violation and raises a single `SchemaError` listing
them all, so a reader port fails loudly with the full picture at once.
"""

import re
from datetime import datetime

import numpy as np
import xarray as xr

from core.geo.naming import names


#: Version stamped on datasets by `Reader.finalize()`
SCHEMA_VERSION = '2.0'

#: Band labels must be scientific names ('B1', 'Oa01', 'VN01', 'coastal_blue'),
#: never wavelengths or bare numbers ('442nm', '353', '13.5'): the wavelength
#: belongs in `cwav`. Letter first, then letters/digits/underscores.
_BAND_LABEL_RE = re.compile(r'[A-Za-z][A-Za-z0-9_]*\Z')

#: Tolerance (nm) for the ascending-cwav check: pure float noise; readers sort
#: by cwav, so a real inversion is orders of magnitude larger than this.
_CWAV_ORDER_TOL = 1e-9

#: Multi-resolution native layout payloads: {Ltoa,Rtoa}_{res}m, one per
#: resolution group (MSI resolution=None).
_NATIVE_PAYLOAD_RE = re.compile(rf'^({str(names.ltoa)}|{str(names.rtoa)})_(\d+)m\Z')

#: Native-layout grid dims: a canonical prefix (bands / rows / columns) plus
#: one common resolution suffix, e.g. bands_10m / y_10m / x_10m.
_NATIVE_GRID_DIM_RE = re.compile(
    rf'^(bands|{str(names.rows)}|{str(names.columns)})_(\d+)m\Z')


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
    # np.unicode_ was removed in NumPy 2; object arrays may hold str labels.
    dtype = np.dtype(dtype)
    return np.issubdtype(dtype, np.str_) or dtype == np.dtype('O')


def validate(ds: xr.Dataset, primary=None, required_vars=()) -> None:
    """
    Validate a v2 reader output against the canonical schema.

    Checks presence/dims/dtype of variables and the mandatory attribute set
    (never data values). Payload units are not mandatory: readers set a `unit`
    attribute only when the product metadata provides one.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check, typically the dataset returned by a reader call.
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
        # The registry dtype is the expected precision; float64 is also
        # accepted: readers whose legacy v1 output was double (PLEIADES, PNEO,
        # SGLI) keep float64 end to end for v1 compatibility (REFACTORING.md).
        if name.dtype and var.dtype not in (np.dtype(name.dtype), np.dtype('float64')):
            errors.append(f'{name}: dtype is {var.dtype}, expected {name.dtype} or float64')

    # --- payload ------------------------------------------------------------
    present = [p for p in PAYLOADS if p in ds]
    # Multi-resolution native layout (MSI resolution=None): one payload per
    # resolution group on its own (bands_{res}m, y_{res}m, x_{res}m) grid.
    # Detected by name; the groups cannot coexist with a canonical payload and
    # must all belong to one family (Ltoa or Rtoa).
    native = sorted(v for v in ds.data_vars if _NATIVE_PAYLOAD_RE.match(str(v)))
    if native:
        if present:
            errors.append(f'native payloads {native} cannot coexist with canonical payloads {present}')
        if len({p.split('_', 1)[0] for p in native}) != 1:
            errors.append(f'native payloads mix payload families: {native}')
        if primary is not None and str(primary) not in native:
            errors.append(f'primary payload {str(primary)!r} is not one of the native payloads {native}')
    else:
        if primary is None:
            if len(present) != 1:
                errors.append(f'expected exactly one of {PAYLOADS} as payload, got {present}')
        else:
            primary = str(primary)
            if primary not in PAYLOADS:
                raise ValueError(f"invalid primary payload {primary!r}, must be one of {PAYLOADS}")
            if primary not in present:
                errors.append(f'missing declared primary payload {primary}')

    for p in [*present, *native]:
        var = ds[p]
        dims = tuple(var.dims)
        if str(names.bands) in dims:
            expected_dims = (str(names.bands),) + grid_dims
        elif _NATIVE_PAYLOAD_RE.match(p):
            # Native group payload: the three dim prefixes must match the
            # canonical ones and carry one common resolution suffix.
            parts = [_NATIVE_GRID_DIM_RE.match(d) for d in dims]
            if len(dims) == 3 and all(parts) \
                    and tuple(x.group(1) for x in parts) == ('bands', str(names.rows), str(names.columns)) \
                    and len({x.group(2) for x in parts}) == 1:
                expected_dims = dims
            else:
                errors.append(
                    f'{p}: dims are {dims}, expected (bands, y, x) or a consistent '
                    f'native grid (bands_{{res}}m, y_{{res}}m, x_{{res}}m)')
                continue
        else:
            # Single-layer payloads (e.g. LST) carry no bands dimension.
            expected_dims = grid_dims
        if dims != expected_dims:
            errors.append(f'{p}: dims are {dims}, expected {expected_dims}')
        # No unit check: payload units are only set when the product metadata
        # states one (readers never invent a unit).

    # Native layouts: each group's band dim carries scientific labels and a
    # per-group cwav table, with the same rules as the canonical bands dim.
    for res in sorted({m.group(2) for m in map(_NATIVE_PAYLOAD_RE.match, native)}):
        bdim, cvar = f'bands_{res}m', f'cwav_{res}m'
        labels = None
        if bdim not in ds:
            errors.append(f'missing coordinate {bdim!r}: band labels are mandatory')
        else:
            bcoord = ds[bdim]
            labels = [str(v) for v in bcoord.values]
            if not _is_string_dtype(bcoord.dtype):
                errors.append(f'{bdim}: coordinate should carry band names (string dtype), got {bcoord.dtype}')
            else:
                bad = [l for l in labels if not _BAND_LABEL_RE.match(l)]
                if bad:
                    errors.append(
                        f'{bdim}: band labels must be scientific names '
                        f'(letter first, then [A-Za-z0-9_]), got wavelength-like or '
                        f'number-only labels: {bad[:8]}')
                if len(set(labels)) != bcoord.size:
                    errors.append(f'{bdim}: band labels are not unique')
        if cvar not in ds:
            errors.append(f'missing required variable {cvar!r}')
        else:
            cw = ds[cvar]
            if tuple(cw.dims) != (bdim,):
                errors.append(f'{cvar}: dims are {tuple(cw.dims)}, expected ({bdim},)')
            elif not np.issubdtype(cw.dtype, np.floating):
                errors.append(f'{cvar}: dtype is {cw.dtype}, expected a floating point type')
            else:
                vals = cw.values
                if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
                    errors.append(f'{cvar}: values must be finite and strictly positive')
                elif labels is not None:
                    inv = np.flatnonzero(np.diff(vals) < -_CWAV_ORDER_TOL)
                    if inv.size:
                        i = int(inv[0])
                        errors.append(
                            f'{cvar}: bands must be sorted by ascending '
                            f'central wavelength; first inversion at index {i} '
                            f'({labels[i]}: {vals[i]:g} nm -> {labels[i + 1]}: '
                            f'{vals[i + 1]:g} nm)')

    # --- bands --------------------------------------------------------------
    if str(names.bands) in ds.dims or str(names.bands) in ds:
        if str(names.bands) not in ds:
            errors.append(f'missing coordinate {names.bands!s}: band labels are mandatory')
        else:
            bcoord = ds[str(names.bands)]
            labels = [str(v) for v in bcoord.values]
            if not _is_string_dtype(bcoord.dtype):
                errors.append(f'{names.bands}: coordinate should carry band names (string dtype), got {bcoord.dtype}')
            else:
                bad = [l for l in labels if not _BAND_LABEL_RE.match(l)]
                if bad:
                    errors.append(
                        f'{names.bands}: band labels must be scientific names '
                        f'(letter first, then [A-Za-z0-9_]), got wavelength-like or '
                        f'number-only labels: {bad[:8]}')
                if len(set(labels)) != bcoord.size:
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
                    else:
                        inv = np.flatnonzero(np.diff(vals) < -_CWAV_ORDER_TOL)
                        if inv.size:
                            i = int(inv[0])
                            errors.append(
                                f'{names.cwav}: bands must be sorted by ascending '
                                f'central wavelength; first inversion at index {i} '
                                f'({labels[i]}: {vals[i]:g} nm -> {labels[i + 1]}: '
                                f'{vals[i + 1]:g} nm)')
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
