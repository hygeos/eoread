# eoread - Read satellite products as xarray Datasets

Read earth observation products as lazy, dask-backed `xarray.Dataset` objects, for integration in processing chains, subsetting, conversion, etc.

Target features:
- one reader class per (sensor, level), sharing a common contract and output schema
- easy input/output, conversion, subsetting and manipulation
- easy processing by blocks
- lazy file access — nothing is computed at construction
- allow processing in parallel with dask
- quality flags decoded on demand, never stored
- read/write NetCDF

## Installation

```bash
hyp add --hygeos eoread
```

## Open a product

A reader is a **class**, one per (sensor, level), grouped by sensor under `eoread.readers`:

```python
from eoread.readers.msi import Level1C_MSI

ds = Level1C_MSI('S2B_MSIL1C_20250320T104639_N0511_R051_T31UDS_20250320T142408.SAFE',
                 chunks=1000, resolution=60)
```

Calling the class reads the product and returns the dataset directly: `ds` is an
`xarray.Dataset` whose variables are lazy dask arrays — they are not read nor
computed until they are accessed.

Every reader shares the base signature

```python
Reader(path, chunks=None, metadata_template=None, verbose=False)
```

and may add reader-specific options as extra keyword arguments with defaults
(e.g. `resolution` on MSI).

Every returned dataset conforms to the canonical output schema and is validated
at read time (stamped with a `schema_version` attribute). Unsupported product
types or versions raise a `ProductError` subclass loudly — never a silent
fallback.

## Supported readers

| Sensor | Level(s) | Module |
|--------|----------|--------|
| Sentinel-2 MSI | L1C, L2A | `eoread.readers.msi` |
| Sentinel-3 OLCI | L1, L2 | `eoread.readers.olci` |
| MERIS | L1 | `eoread.readers.meris` |
| Landsat 8/9 OLI | L1TP | `eoread.readers.oli` |
| MODIS | L1B | `eoread.readers.modis` |
| ECOSTRESS | L1B, L1C, L2 | `eoread.readers.ecostress` |
| PACE OCI | L1B | `eoread.readers.pace` |
| VENµS | L1C, L2A | `eoread.readers.venus` |
| SGLI | L1B | `eoread.readers.sgli` |
| HYPSO-1 | L1 | `eoread.readers.hypso` |
| GOES-NG | L1 | `eoread.readers.goesng` |
| GEOSAT (DE-1/DE-2) | L1 | `eoread.readers.geosat` |
| Pleiades / Pleiades Neo | L1 | `eoread.readers.pleiades`, `eoread.readers.pleiades_neo` |
| PlanetScope | L1 | `eoread.readers.planetscope` |

`eoread.reader.discover_readers()` lists every reader class programmatically.

## Subsetting

xarray Datasets allow easy subsetting:

```python
# product subsetting based on row and column indices (canonical dim names)
sub = ds.isel(y=slice(500, 600), x=slice(400, 500))
```

With geographical coordinates:

```python
sub = ds.sel(lat=slice(45.5, 46.0), lon=slice(3.0, 3.5))
```

The `core` package provides helpers for geographic subsetting:

```python
from core.tools import sub_rect, sub_pt

# based on a range of lat/lon
sub = sub_rect(ds, lat_min, lon_min, lat_max, lon_max)

# based on a center and a radius (km)
sub = sub_pt(ds, pt_lat, pt_lon, rad)
```

## Quality flags

No pre-decoded flag masks are stored in the dataset. Flags are decoded on
demand from the raw flag layers by the reader class:

```python
from eoread.flags import GenericFlags

# lazy boolean mask, e.g. invalid pixels for MSI
mask = Level1C_MSI.getflag(ds, GenericFlags.L1_INVALID)
```

Readers supporting flags advertise themselves through the
`ds.attrs['_flag_reader']` attribute. To pack several flags into a single
variable for blockwise processing:

```python
from eoread.flags import FlagsInit, GenericFlags

init = FlagsInit(
    flags={GenericFlags.LAND: 1 << 0, GenericFlags.CLOUD: 1 << 1},
    dtype='uint16',
    flag_reader=ds.attrs['_flag_reader'],
)
ds = init.map_blocks(ds)   # adds a packed 'flags' variable
```

Flag support varies by reader; `getflag` raises `ValueError` for an
unsupported flag.

## Metadata

By default readers keep the full product metadata in `ds.attrs['metadata']`.
Pass `metadata_template` to keep only selected branches:

```python
ds = Level1C_MSI('product.SAFE',
                 metadata_template=[['General_Info', 'Product_Info'],
                                    ['Product_Identifier']])
```

## Output products

Writing to NetCDF is supported by the `xarray.Dataset.to_netcdf` method.

A helper function is provided by `core.files.to_netcdf`, which provides
features like automatic file naming, temporary files and compression.

## Chunking

For efficient processing:

- Do NOT chunk and then sel. This leads to very poor efficiency. Subset first, and chunk your data as late as possible.
- See also http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance

## Migrating from v1

The pre-v2 readers (`eoread.legacy.*`, including the `Level1()`/`Level2()`
auto-detection functions) were removed in 2.0. To map a v2 dataset back to the
v1 layout:

```python
from eoread.compat import v1
ds_v1 = v1(ds)
```

A few sensors carry documented clean breaks where v2 values intentionally
differ from v1 (see `eoread.compat.CLEAN_BREAKS`).

## Tests

Uses pytest, on real products only; tests skip cleanly when a sample product
is not available:

    $ pixi run gate     # no-data gate (import smoke + collection)
    $ pytest tests/
