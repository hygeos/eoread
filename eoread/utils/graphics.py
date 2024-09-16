from matplotlib import pyplot as plt
import xarray as xr


def plot_srf(srf: xr.Dataset):
    """
    Plot a SRF Dataset
    """
    plt.figure()
    for iband in srf.data_vars:
        srf[iband].plot(label=iband)
        for coord in srf[iband].coords:
            assert "units" in srf[coord].attrs
    plt.title(srf.desc)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel("wavelength")
    plt.ylabel("SRF")
    plt.grid(True)
