#!/usr/bin/env python3
"""
Radiometry helpers shared by v2 readers.

Per-sensor calibration math (quantization, offset/gain scaling, dark current...)
stays in the reader itself; this module holds only formulas that are genuinely
common across sensors.
"""

import numpy as np
import xarray as xr


def reflectance_from_radiance(ltoa: xr.DataArray, f0, mus) -> xr.DataArray:
    """
    Top-of-atmosphere reflectance from top-of-atmosphere radiance.

        Rtoa = pi * Ltoa * mus / F0

    Parameters
    ----------
    ltoa : DataArray
        TOA spectral radiance.
    f0 : array-like, broadcastable to ltoa
        Solar spectral irradiance at the top of the atmosphere (same units),
        per band.
    mus : array-like, broadcastable to ltoa
        Cosine of the sun zenith angle (per pixel, or scalar).

    Returns
    -------
    DataArray (float32) with ltoa's dims and coords, unclipped.
    """
    rtoa = np.pi * ltoa * mus / f0
    return rtoa.astype('float32')
