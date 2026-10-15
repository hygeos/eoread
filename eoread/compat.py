#!/usr/bin/env python3
"""
v1 compatibility transform (see REFACTORING.md, §9).

All v2 readers emit the same canonical schema, so a single transform serves
every sensor: `v1(ds)` maps a canonical dataset back to the legacy v1 layout
(variable renames and reorganization), letting downstream code keep working
during the transition.

Rule (locked): compat is supported when the v1→v2 difference of a reader is a
pure rename/reorganization. When the legacy output carried different *values*
for a variable, that reader gets a documented clean break instead — record it
in `CLEAN_BREAKS` with an explanation.
"""

from core.geo.naming import names


#: Per-sensor v2→v1 renames, applied when ds.attrs['sensor'] matches.
#: Only entries whose name actually changed between v1 and v2.
_V1_RENAMES = {
    # MSI L1C: the payload is TOA radiance; v1 misnamed it Rtoa (see REFACTORING.md §5).
    'MSI': {str(names.ltoa): str(names.rtoa)},
}

# MSI L2A band labels: the legacy reader built its band lists with blind pops
# (`cwvl.pop(7); cwvl.pop(10)`) that mislabeled B11 as 'B10'. v2 builds the
# band table from the files actually read, so `bands`/`cwav` coordinates are
# correct. This is a documented deviation (REFACTORING.md decision log), not a
# CLEAN_BREAKS entry: it lives in coordinate *values*, which this transform
# deliberately does not touch.
CLEAN_BREAKS = {
    # OLI L1TP: v2's Rtoa drops the legacy /cos(sza) division — the USGS C2
    # reflectance factors already fold in the solar geometry, so dividing
    # again was a double correction (REFACTORING.md decision log). The pan
    # band B8 also moved from separate 'rows_pan'/'columns_pan' dims onto the
    # single bands grid, and the azimuth angles are folded to the unsigned
    # [0, 180] canonical range instead of the signed file values. v1() cannot
    # reproduce the legacy values.
    'OLI': ('v2 Rtoa differs from the legacy values (the /cos(sza) division is '
            'removed: the USGS C2 reflectance factors already include the solar '
            'correction), band B8 now shares the single bands grid, and the '
            'azimuth angles are unsigned [0, 180] instead of the signed file '
            'values'),
}


def v1(ds):
    """
    Map a canonical v2 dataset back to the legacy v1 layout.

    The input dataset is not modified; a new dataset is returned.

    Raises
    ------
    NotImplementedError
        If the sensor of `ds` has a documented clean break (see `CLEAN_BREAKS`).
    ValueError
        If a rename would collide with an existing variable.
    """
    sensor = ds.attrs.get(str(names.sensor))
    if sensor in CLEAN_BREAKS:
        raise NotImplementedError(CLEAN_BREAKS[sensor])

    mapping = {}
    for new, old in _V1_RENAMES.get(sensor, {}).items():
        if new not in ds:
            continue
        if old in ds and old != new:
            raise ValueError(f'v1 compat for {sensor}: cannot rename {new!r} to {old!r}, already present')
        mapping[new] = old

    ds = ds.rename(mapping)
    if 'schema_version' in ds.attrs:
        # v1 datasets did not carry a schema version
        ds.attrs = {k: v for k, v in ds.attrs.items() if k != 'schema_version'}
    return ds
