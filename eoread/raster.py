#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
try:
    import gdal
except ModuleNotFoundError:
    gdal = None



class ArrayLike_GDAL:
    """
    Read a file (eg GeoTIFF) with Gdal as an array-like
    """
    def __init__(self, filename: str):
        assert Path(filename).exists()
        if gdal is None:
            raise Exception('Error, gdal is not available.')

        dset = gdal.Open(filename)
        band = dset.GetRasterBand(1)
        self.width = band.XSize
        self.height = band.YSize
        self.shape = (self.height, self.width)
        self.ndim = len(self.shape)
        self.filename = filename
        self.dtype = band.ReadAsArray(
            win_xsize=0,
            win_ysize=0,
        ).dtype

    def __getitem__(self, keys: list):
        ystart = int(keys[0].start) if keys[0].start is not None else 0
        xstart = int(keys[1].start) if keys[1].start is not None else 0
        ystop = int(keys[0].stop) if keys[0].stop is not None else self.shape[0]
        xstop = int(keys[1].stop) if keys[1].stop is not None else self.shape[1]

        dset = gdal.Open(self.filename)   # NOTE: we have to re-open the file each time to avoid a segfault
        band = dset.GetRasterBand(1)
        data = band.ReadAsArray(  # NOTE: step is not supported by gdal, have to apply a posteriori
            xoff=xstart,
            yoff=ystart,
            win_xsize=xstop - xstart,
            win_ysize=ystop - ystart,
            )[::keys[0].step, ::keys[1].step].astype(self.dtype)

        return data
