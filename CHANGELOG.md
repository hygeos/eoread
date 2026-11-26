# Changelog

## v2.0.0 (2026-11-26)

Complete rewrite of the readers around a common contract. **Breaking release.**

### Breaking changes
- One reader class per (sensor, level) under `eoread.readers.*`, named after the
  product level (e.g. `Level1C_MSI`, `Level1_OLCI`). Calling the class returns
  the `xarray.Dataset` directly, with the shared signature
  `Reader(path, chunks=None, metadata_template=None, verbose=False)`.
- All readers emit a canonical output schema, validated at read time and stamped
  with a `schema_version` attribute. Unsupported products raise a `ProductError`
  subclass instead of falling back silently.
- Band labels are scientific names, sorted by ascending central wavelength.
- Flag readers are merged into the readers: flags are decoded on demand
  (`Reader.getflag`), never stored in the dataset.
- Legacy v1 code removed (`eoread.legacy.*`, `Level1()`/`Level2()`
  auto-detection). `eoread.compat.v1(ds)` maps a v2 dataset back to the v1
  layout; intentional value differences are listed in `eoread.compat.CLEAN_BREAKS`.
- Dependency on `core` replaced by `hygeos-core` (PyPI, imported as `hygeos_core`).

### Readers
- Ported to v2: MSI, OLCI (L1 EFR/ERR, L2 WFR/WRR), MERIS (N1 and SAFE, FR/RR),
  Landsat OLI, MODIS, ECOSTRESS (L1B, L1C, L2), PACE, VENµS (L1C, L2A), SGLI,
  HICO, PRISMA, HYPSO-1, GOES-NG, and the DIMAP readers (GEOSAT, Pleiades,
  Pleiades Neo, PlanetScope).
- NASA Level-1C reader, `makeL1C` transform (incl. VIIRS JPSS-2) and ancillary
  providers.
- MSI: `resolution=None` for the native multi-resolution layout; `angle_band`
  option for per-band viewing geometry.
- MERIS N1: `LAND` flag. PACE: terrain height provided as altitude.
- OLCI: invalid detector indices masked in the per-pixel spectral gather.

### Tooling
- Sample-product registry; `Reader.get_sample` downloads samples on demand via SAND.
- No-data test gate (`pixi run gate`) and a v1 vs v2 benchmark harness.
- HDF5 stack from conda-forge; documentation updated for v2.

## v1.0.0 (2026-11-19)

Consolidation of the v1 readers (since v0.4.0).

### Added
- New readers: Sentinel-2 MSI L2A, PRISMA L1, ISS HICO, MERIS 4th reprocessing
  (SAFE format); Sentinel-2C support.
- Flags readers for every sensor, with generic flags (`L1_INVALID`,
  `L1_DEGRADED`, `LAND`, …) and a non-strict mode for `FlagsInit`.
- `_srf_getter` attribute on readers (MSI, OLI incl. panchromatic, NASA sensors).
- MSI: embedded ECMWF ancillary data, per-detector viewing angles,
  `angle_band` option for per-band viewing geometry.
- `makeL1C`: major update, `ocssw_version` and `ocssw_tags` parameters.
- `Ancillary_NASA`: TCWV, picklable.
- Plotting utilities (`xrimshow`, spectrum plots), crop helper, several angle
  interpolation methods, option to mute debugging messages.
- Logging in each reader; documentation migrated to quartodoc with NumPy
  docstrings and a GitHub Action.

### Changed
- New representation of the bands dimension across all readers.
- `spatial_resample`: lazy tie-point interpolation per output chunk.
- Lazy, chunked reading for MERIS N1 and OLI lat/lon.
- MODIS reader uses netCDF4 instead of the HDF4 module; Landsat module renamed
  `landsat_oli` → `oli`; generic `QUALITY` flag renamed `L1_DEGRADED`.
- Modules superseded by SAND/HARP removed (deprecated shims kept for
  back-compatibility); SAND no longer a hard dependency.

### Fixed
- Geolocation and orientation fixes (NASA sensors, OLI, ECOSTRESS).
- MSI angle interpolation and band order; OLCI chunking and LAND flag.
- OLI TOA reflectance normalization; MODIS reading with netCDF4.
- PACE band order, land flag and altitude; PRISMA Ltoa units; SGLI geometry
  scaling, invalid radiances and flags; VENµS angles and flags; MERIS units,
  F0 and meteo variables; ECOSTRESS BT computation and invalid flag.
