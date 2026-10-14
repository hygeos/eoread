#!/usr/bin/env python3
"""
Read a Sentinel-2 MSI product with the v2 reader and print the dataset.

Usage:
    pixi run python scripts/read_msi.py              # L1C sample (LEVEL1_MSI)
    pixi run python scripts/read_msi.py <product>    # any .SAFE path
"""
from eoread.readers.msi import Level1C_MSI
from core.env import getdir

reader = Level1C_MSI(getdir('LEVEL1_MSI'), resolution=60)
print(reader.ds)
