
# Ancillary Module

## Description


##### This module allows for easy access to ancillary data, and seamless transition from the providers:
  * **MERRA2** (NASA)
  * **ERA5** (Copernicus CDS)
  * **CAMS** (Copernicus ADS)

The interfaces are unified. The variables names used for querying are standardized.
This allow for seamless switch to another source, provided that the variable also exists in the other sources.

Name convention is specified in the nomenclature.csv file, which can be overriden locally.

## Table of Content
- [Description](#Description)
- [Features](#Features)
- [Requirements](#Requirements)
- [Usage](#usage)
- [License](#license)

## Features

* Always download a **full day of data** (may be subject to change in futur versions) with a specified file nomenclature.
This allows the module to find the local data first, if it exists, and return it instead of downloading.

* Uses **filegen** tool to download to a **temporary file** first, and copy the whole file after the download has completed

* The module can define **computable variables** to fill the gaps between the providers, they should be defined in the nomenclature CSV file with a name starting with **'#'** to differenciate them from provided variables.
The module also document this process by writing to the **attrs['history']** attribute of the DataArray.


## Requirements

#### MERRA2:
    
Entry for NASA server in .netrc file, present in $HOME:

    machine urs.earthdata.nasa.gov 
    login username 
    password mypassword
    
Require an account at NASA's URS: https://urs.earthdata.nasa.gov/

#### ERA5:
Entry for the CDS in .cdsapirc file, present in $HOME:

    url: https://cds.climate.copernicus.eu/api/v2
    key: my-cds-key

Require an account at the CDS from Copernicus: https://cds.climate.copernicus.eu/

#### CAMS:
Entry for the ADS in .cdsapirc file, present in $HOME:

    ads.url: https://ads.atmosphere.copernicus.eu/api/v2
    ads.key: my-ads-key

Require an account at the ADS from Copernicus: https://ads.atmosphere.copernicus.eu/


## Usage:


### Import:
~~~python
    from eoread.ancillary import MERRA2
    from eoread.ancillary import ERA5
    from eoread.ancillary import CAMS
    
    from datetime import date, datetime
~~~
    
### Instantiate:
~~~python
cams = CAMS(
    model=CAMS.models.global_atmospheric_composition_forecast,
    directory=Path(download_dir)
    )

merra = MERRA2(
    model=MERRA2.models.M2T1NXSLV,
    directory=Path(download_dir)
    )

era5 = ERA5(
    model=ERA5.models.reanalysis_single_level,
    directory=Path(download_dir)
    )
~~~

The **`model`** parameter allow to select between different models such as single level and pressure levels in era5.
The possible models are gathered in the **`.models`** attributes of the **classes**, allowing for **autocompletion** to work.

The directory parameter allow to choose where the file will be downloaded, and read.

**Optionnal parameters:**

  * **nomenclature_file:** override nomenclature filepath used for name standardization
  * **offline:** disable download if True, can only read existing file, will raise if not found
  * **verbose:** enable or disable verbosity
  * **no_std:** disable name standardization if True, the returned variables will be named as provided

### Query Data:
  
~~~python
variables = ['total_column_water_vapor']

# get data interpolated on time
ds = era5.get(variables, dt=datetime(2013, 11, 30, 13, 35))

# get full day of data
ds = era5.get_day(variables, dt=date(2013, 11, 30,))

# get severals days of data
ds = era5.get_range(variables, date_start=date(2020, 3, 21), date_end=date(2020, 3, 23)))
~~~

**Optionnal parameters:**

  * **area:** regionalize the data query to specified lats lons (will round to outer integers)
  
example:

~~~python
# query data for lat E [15.3, .., 17.5] and lon E [-11.3, .., -10.1]
ds = era5.get_day(variables, dt=date(2013, 11, 30),   area = [17.5, -11.3, 15.3, -10.1])
~~~

### Switching provider:

Because the interfaces are unified and since **ERA5**, **CAMS** and **MERRA2** all provide the variable **'total_column_water_vapor'** (perhaps named differently), we can query this variable from all of them, in the exact same manner.

~~~python
# get full day of data from era5
ds = era5.get_day(variables, dt=date(2013, 11, 30))

# get full day of data from cams
ds = cams.get_day(variables, dt=date(2013, 11, 30))

# get full day of data from merra2
ds = merra2.get_day(variables, dt=date(2013, 11, 30))
 ~~~


### Pre-Processing

Setting the parameter **`offline`** to **`True`** in the constructor disables the download capabilities of the classes. They will therefore only try to get the files locally, and will raise an error if it cannot be found. This can be usefull to write a pre-processor to check that every file needed is already present locally.

## Author and Contacts

##### HYGEOS, Joackim Orcière, joackim.orciere@hygeos.com

