#!/usr/bin/env python3
"""
Pleiades-Neo DIMAP strip reader, level L1 (reflectance).

    >>> from eoread.readers.pleiades_neo import Level1_PNEO         # doctest: +SKIP
    >>> ds = Level1_PNEO('WO_000194880_1_1_SAL24178541-1_ACQ_...')  # doctest: +SKIP

Shares the Pleiades family base (`eoread.readers.pleiades._DimapBase`); this
module only carries the PNEO-specific differences.

Differences from the legacy v1 readers (documented deviations, see the
REFACTORING.md decision log):

- The output is south-up: row 0 is the southern edge of the strip (canonical
  orientation). Legacy outputs were north-up.
- All six spectral bands are kept and sorted by central wavelength; the
  legacy reader silently dropped the red-edge and blue bands.
- Nodata pixels (DN == 0 per the per-file Special_Value tables) become NaN;
  the legacy rioxarray open kept them as zero data (masked=False default).
- No dtype is ever coerced: each variable keeps whatever type its xarray
  read and arithmetic naturally produce (in practice float64 for the
  payload, angles and lat/lon, matching the legacy reader).
- `datetime` uses the ISO-8601 'T' separator (legacy used a space between
  date and time).
"""

from pathlib import Path

import numpy as np
import xarray as xr

from core.geo.naming import names

from eoread.readers.pleiades import _DimapBase
from eoread.tools import _num


class Level1_PNEO(_DimapBase):
    """Pleiades-Neo multi-spectral L1 reflectance strips (PMS mode)."""

    sensor = 'PNEO'
    sample_products = ('PNEO_MS',)

    def _open_payload(self, strip: Path, dim: dict) -> tuple[xr.DataArray, list]:
        # The MS product spans several JP2 files; each Data_Files entry maps a
        # file to its bands through Raster_Index (per-file BAND_INDEX restarts
        # at 1). Files are concatenated in document order.
        files = dim['Raster_Data']['Data_Access']['Data_Files']
        if isinstance(files, dict):
            files = [files]
        arrays, bands = [], []
        for entry in files:
            href = str(entry['Data_File']['DATA_FILE_PATH']['attributes']['href'])
            indices = entry['Raster_Display']['Raster_Index_List']['Raster_Index']
            if isinstance(indices, dict):
                indices = [indices]
            order = sorted(indices, key=lambda r: int(r['BAND_INDEX']))
            ids = [str(r['BAND_ID']) for r in order]
            dn = self._lazy_open_jp2(strip / href)
            dn = self._mask_nodata(dn, self._nodata_values(entry['Raster_Display']))
            arrays.append(dn.assign_coords({str(names.bands): ids}))
            bands.extend(ids)
        return xr.concat(arrays, dim=str(names.bands)), bands

    def _band_table(self, dim: dict, bands: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bml = (dim['Radiometric_Data']['Radiometric_Calibration']
               ['Instrument_Calibration']['Band_Measurement_List'])
        spec = {s['BAND_ID']: s for s in bml['Band_Spectral_Range']}
        refl = {s['BAND_ID']: s for s in bml['Band_Reflectance']}
        # FWHM MIN/MAX are already in nm (unlike Pleiades' um range).
        cwav = np.array([(float(_num(spec[b]['FWHM']['MIN']))
                          + float(_num(spec[b]['FWHM']['MAX']))) / 2
                         for b in bands], dtype='float64')
        gains = np.array([float(_num(refl[b]['GAIN'])) for b in bands])
        bias = np.array([_num(refl[b]['BIAS'], default=0.0) for b in bands])
        return cwav, gains, bias
