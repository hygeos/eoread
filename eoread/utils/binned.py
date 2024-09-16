import xarray as xr
import numpy as np


"""
Interface with binned L3 products (GSFC)

See https://169.154.128.44/docs/technical/ocean_level-3_binned_data_products.pdf
"""


def ncols(neq):
    """
    returns an array of the number of valid columns per row in
    sinusoidal projection with neq bins at equator
    """
    return np.round(np.sin(np.linspace(0, 1, neq+1)[1::2]*np.pi)*neq).astype('int')


def latlon2bin_sinu(lat, lon, neq, lon0=0.):
    """
    lat, lon to bin index in sin grid with neq bins at equator
    """
    nc = ncols(neq)

    # index of first bin of each row
    start_num = np.concatenate(([0], np.cumsum(nc)[:-1]))

    ilat = ((neq/2)*(lat+90.)/180.).astype('int')
    ilon = np.minimum((nc[ilat]*(lon+180-lon0 % 360)/360.).astype('int'), nc[ilat])

    return ilon + start_num[ilat]


class Binner:
    """
    Accululate values in a (flattened) sinusoidal grid
    """
    def __init__(self, neq):
        nc = ncols(neq)
        self.nbins = np.sum(nc)
        self.sums = np.zeros(self.nbins, dtype='float64')
        self.counts = np.zeros(self.nbins, dtype='int32')
        self.neq = neq

    def add(self, values, lat, lon):
        ibins = latlon2bin_sinu(lat, lon, self.neq)
        self.sums += np.bincount(
            ibins,
            weights=values,
            minlength=self.nbins)
        self.counts += np.bincount(
            ibins,
            minlength=self.nbins)

    def values(self):
        return self.sums/self.counts


def read_binned(filename, varname, groupname='level-3_binned_data'):
    """
    Opens a binned product in sinusoidal projection as a 2-dim array

    returns: data, lat, lon
    """
    ds = xr.open_dataset(
        filename,
        group=groupname,
        )
    neq = 2*len(ds.BinIndex)
    bin_num = ds.BinList.data['bin_num']
    extent = ds.BinIndex.data['extent']
    data = ds[varname].data['sum']/ds.BinList.data['weights']

    return to_2dim(data, neq, bin_num, extent)


def to_2dim(data, neq, bin_num=None, extent=None):
    """
    Returns a 2-dimensional array from flattened data in sinusoidal grid
    """
    H = neq//2
    W = neq
    nc = ncols(W)
    nbins = nc.sum()
    if bin_num is None:
        bin_num = np.arange(nbins) + 1
    if extent is None:
        extent = nc

    # index of first bin of each row
    start_num = np.concatenate(([0], np.cumsum(nc)[:-1]))
    ilat = np.repeat(np.arange(H), extent)
    icol = bin_num - np.repeat(start_num, extent) - 1

    off = (W-1-np.repeat(nc, extent))//2

    lat_flat = ((ilat+1-.5)*180./H) - 90.
    lon_flat = (360. * (icol+.5)/np.repeat(nc, extent)) - 180.

    reprojected = np.zeros((H, W)) + np.NaN
    reprojected[H-1-ilat, icol + off] = data
    lat = np.zeros((H, W)) + np.NaN
    lat[H-1-ilat, icol + off] = lat_flat
    lon = np.zeros((H, W)) + np.NaN
    lon[H-1-ilat, icol + off] = lon_flat

    return reprojected, lat, lon
