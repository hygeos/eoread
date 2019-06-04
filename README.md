# eoread - Read satellite products as xarray Datasets

Read earth observation products as xarray Datasets, for integration in processing chains, subsetting, conversion, etc.

Target features:

    - easy input/output, conversion and manipulation
    - easy processing by blocks
    - lazy file access
    - allow processing in parallel (with dask ?)
    - easy subsetting, including based on lat, lon window
    - geometric inclusion tests
    - read/write NetCDF or ASCII
    - use attributes for each dataset

Testing: using pytest
