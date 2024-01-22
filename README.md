# eoread - Read satellite products as xarray Datasets

Read earth observation products as xarray Datasets, for integration in processing chains, subsetting, conversion, etc.

Target features:
- easy input/output, conversion, subsetting and manipulation
- easy processing by blocks
- lazy file access
- allow processing in parallel with dask
- geometric inclusion tests
- read/write NetCDF or ASCII
- use attributes for each dataset

## Open a product

```python
from eoread.olci import Level1_OLCI
l1 = Level1_OLCI('S3A_OL_1_EFR____20190430T094655_[...].SEN3/')
```

`l1` is a `xarray.Dataset` containing all variables and attributes. All variables are lazy dask arrays: they are not read nor computed until they are accessed.

OLCI level1 contains Ltoa. To add Rtoa:
```python
from eoread import eo
eo.init_Rtoa(l1)
```

## Subsetting

xarray Datasets allow easy subsetting:
```python
# product subsetting based on rows and columns indices
sub = l1.isel(y=slice(500, 600),
              x=slice(400, 500))
```

With geographical coordinates:
```python
# based on a range of lat/lon
sub = sub_rect(ds, lat_min, lon_min, lat_max, lon_max)
```

```python
# based on a center and a radius
sub = sub_pt(ds, pt_lat, pt_lon, rad)   # rad is the radius in km
```


## Processing

Use xr.apply_ufunc to apply a universal function (in the numpy sense)
to each block of a Dataset or DataArray.

## Output products

Writing to NetCDF is supported by the `xarray.Dataset.to_netcdf` method.

A helper function is provided by `eo.to_netcdf`, which provided features like automatic file
naming, temporary files and compression.


## Chunking

For efficient processing:

- Chunks should be aligned with NetCDF chunks (`netCDF4.Dataset(f).variables[v].chunking()`)
- Do NOT chunk and then sel. This leads to very poor efficiency. Chunk your data as late as possible.
- See also http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance


## Tests

Uses pytest:

    $ pytest
