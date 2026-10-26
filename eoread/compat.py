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

Per-sensor status (Phase-6 final pass; rationale in REFACTORING.md §9/§10):

- MSI: rename ``Ltoa`` → ``Rtoa`` (v1 misnamed TOA radiance) + schema_version
  strip. Documented deviations, NOT breaks: the L2A band labels are corrected
  where legacy's blind pops mislabeled B11 as 'B10' (coordinate values — this
  transform never touches coordinates); lat/lon stored float32 per the naming
  registry; tie-point angles differ from legacy within the documented
  interpolation tolerance (max |Δ| ~1e-3°, localized to the steepest azimuth
  gradient pocket).
- ECOSTRESS: identity + schema_version strip (all payloads kept their v1
  names). Documented deviation, NOT a break: v2 is south-up where legacy was
  north-up — v2 row i carries legacy row ny-1-i (pinned by test_l1c_v1_compat).
  The plain-HDF5 L1B/L2 family has no legacy reference at all.
- OLCI: identity + schema_version strip — no documented clean break.
- MODIS: clean break (see ``CLEAN_BREAKS``) — radiance formula fix.
- OLI: clean break (see ``CLEAN_BREAKS``) — Rtoa double solar correction,
  single bands grid, unsigned azimuths.
- MERIS: identity + schema_version strip (all payloads kept their v1 names;
  Ltoa and every full-grid field are bit-identical to legacy at mirrored rows).
  Documented deviations, NOT breaks: south-up flip (v2 row i carries legacy
  row ny-1-i); ``F0``/``wav`` are always emitted where legacy gated them behind
  ``read_auxdata=False`` — a superset of the default legacy output; resolution
  comes from the SPH COLUMN_SPACING field instead of legacy's hardcoded 300 m;
  ``shortname`` and ``crs`` attributes were added; azimuth angles keep the
  product's absolute [0, 360) convention (registry declares [0, 180] — a
  documented registry inconsistency, see the OLCI/MSI entries below).
- VENUS: identity + schema_version strip (all payloads kept their v1 names;
  L1C payload values are bit-identical to legacy). Documented deviations, NOT
  breaks: south-up flip (v2 row i carries legacy row ny-1-i); new canonical
  tie-point angles where legacy computed none (raw 2x2 grids kept as ``*_tie``);
  SOL/VIE kept raw on their native tie grids (VIE is no longer reindexed onto
  SOL's smaller grid); L2A product nodata (-10000) becomes NaN where legacy
  kept -10.0 after calibration; L2A ATB calibrated with the MTD quantification
  factors where legacy kept raw counts; L2A PIX_XS assigned positionally (the
  legacy int-vs-string band alignment produced an all-NaN variable); lat/lon
  from a proper per-block UTM transform instead of legacy's flipped corner
  linear interpolation.
- PACE: identity + schema_version strip (payload kept its v1 name ``Rtoa``;
  values are bit-identical to legacy at the pinned window). Documented
  deviations, NOT breaks: per-band chunking where legacy read one all-bands
  block; flags returned as boolean masks where legacy returned raw values —
  ``LAND`` repairs a latent legacy TypeError (``~water`` on the float32
  watermask); ``shortname``/``crs`` attributes were added; ``resolution``
  keeps legacy's hardcoded 1200 m (no data-driven spacing in the product);
  azimuth angles keep the product's signed relative [-180, 180] convention
  (registry declares [0, 180] — a documented registry inconsistency, see the
  OLCI/MSI entries above). No orientation change: the product is natively
  south-up and legacy kept it that way.
- SGLI: identity + schema_version strip (all payloads kept their v1 names;
  ``water``/``quality_flag`` are bit-identical to legacy at mirrored rows).
  Documented deviations, NOT breaks: the output is south-up where legacy
  followed the file's line order — for a north-up-stored granule (the sample)
  v2 row i carries legacy row ny-1-i; the tie fields carry the full affine
  ``Slope*x + Offset`` where legacy applied ``(1+Slope)*x`` (lat/lon doubled,
  centidegree angles inflated ~101x), so geometry and Rtoa (which divides by
  cos of sza) differ from legacy — a confirmed bug fix pinned against the
  product's raw tie data / counts; float32 geometry per the naming registry;
  ``LAND`` = ``water >= 50`` where legacy's ``~water`` on the uint8
  land-percentage was True on every pixel; ``L1_INVALID`` is the negation of
  QA_flag bit 0 where legacy used the raw value (the file's bit description
  says 1 = "Integrity", so the legacy test flagged the sound ~86% of the
  sample scene);
  ``add_ancillary_data`` dropped (instrument telemetry with no consumer);
  ``shortname``/``crs`` attributes were added; ``datetime`` is normalized to
  ISO from the product's non-ISO ``Scene_center_time`` (same instant).
- PLEIADES: identity + schema_version strip (payload kept its v1 name
  ``Rtoa``). Documented deviations, NOT breaks: south-up flip (v2 row i carries
  legacy row ny-1-i); nodata DN==0 becomes NaN where the legacy rioxarray open
  kept it as zero data; payload/angles/lat-lon are float32 end to end where
  legacy mixed float64 intermediates.
- PNEO: identity + schema_version strip (payload kept its v1 name ``Rtoa``).
  Documented deviations, NOT breaks: south-up flip (v2 row i carries legacy row
  ny-1-i); all six spectral bands are kept (sorted by central wavelength) where
  the legacy reader silently dropped the red-edge and deep-blue bands; nodata
  DN==0 becomes NaN where legacy kept it as zero data; azimuth angles keep the
  product's unsigned [0, 360) convention (registry declares [0, 180] — a
  documented registry inconsistency, see OLCI/MSI/MERIS); payload/angles/lat-lon
  are float32 end to end where legacy mixed float64 intermediates.
- GEOSAT: identity + schema_version strip (payload kept its v1 name ``Ltoa``;
  one reader class serves the DE-1 and DE-2 variants). Documented deviations,
  NOT breaks: south-up flip (v2 row i carries legacy row ny-1-i); GEOSAT-2
  nodata DN==0 becomes NaN where legacy kept it as zero data (the calibration
  bias survives in the legacy output there); ``datetime`` is the full DE-2
  START_TIME where legacy emitted a date-only value; ``cwav`` is float64 where
  legacy was int32.
- PLANETSCOPE: identity + schema_version strip (payload kept its v1 name
  ``Rtoa``). Documented deviations, NOT breaks: south-up flip (v2 row i carries
  legacy row ny-1-i); nodata DN==0 becomes NaN where legacy kept it as zero
  data; ``cwav`` is float64 where legacy was float32.
- GOESNG: identity + schema_version strip. No v1 reference exists: the legacy
  function is unusable (it references undefined ``naming``/``eo``/``flags``
  globals and never reaches a result), so the v2 port is verified against the
  raw NetCDF files instead. Documented deviations from the legacy intent:
  south-up flip; string band labels + ``cwav`` (legacy used integer
  wavelengths and a ``wav`` variable); float32 payload/geometry; the optional
  NWC/GEO cloud mask is an explicit ``cloud`` layer decoded through
  ``getflag(CLOUD)`` instead of a packed ``flags`` variable.
- HYPSO: identity + schema_version strip (payload kept its v1 name ``Ltoa``).
  No usable v1 reference exists either: the legacy function leaves the payload
  uint16, keeps the north-up orientation and raises on a bands coordinate that
  the product does not carry. Documented deviations: south-up flip; ``Ltoa``
  float32 (integer radiance is preserved exactly by the cast); 0-based string
  band labels + ``cwav`` from the file's wavelength table; ``datetime``
  normalized to ISO from ``date_aquired``; sensor/platform taken from the
  modern ``instrument_id``/``sat_id`` fields when present (the legacy values
  otherwise). Both geolocation layouts are read: the old ``navigation`` group
  and the current hypso-package ``geometry`` group.
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
    # MODIS L1B: v2's Ltoa uses the PUG M1054E formula scale * (raw - offset);
    # the legacy reader computed scale * raw + offset, so the legacy radiance
    # values are not reproducible from the v2 output.
    'MODIS': ('v2 Ltoa differs from the legacy values (the radiance formula is '
              'scale * (raw - offset) per PUG M1054E; the legacy reader used '
              'scale * raw + offset)'),
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
