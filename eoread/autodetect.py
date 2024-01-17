#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic product loader with autodetection
"""

from pathlib import Path
import xarray as xr
import re


def Level1(path: Path, **kwargs) -> xr.Dataset:
    if re.match("^S3[AB]_OL_1_EFR____.*", path.name):
        from eoread.olci import Level1_OLCI

        return Level1_OLCI(path, **kwargs)

    elif re.match("^S2[AB]_MSIL1C_.*", path.name):
        from eoread.msi import Level1_MSI

        return Level1_MSI(path, **kwargs)

    else:
        raise ValueError(f"Could not detect Level1 type for {path.name}")


def Level2(path: Path, **kwargs) -> xr.Dataset:
    if re.match("^S3[AB]_OL_2_WFR____.*", path.name):
        from eoread.olci import Level2_OLCI

        return Level2_OLCI(path, **kwargs)

    else:
        raise ValueError(f"Could not detect Level1 type for {path.name}")
