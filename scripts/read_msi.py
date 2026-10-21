#!/usr/bin/env python3
"""
Read a Sentinel-2 MSI product with the v2 reader and print the dataset.

Usage:
    pixi run python scripts/read_msi.py              # registered MSI_L1C sample
    pixi run python scripts/read_msi.py <product>    # any .SAFE path
"""
import sys

from eoread.readers.msi import Level1C_MSI

path = sys.argv[1] if len(sys.argv) > 1 else Level1C_MSI.get_sample()
ds = Level1C_MSI(path, resolution=60)
print(ds)
