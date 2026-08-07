from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import xarray as xr
from core import tools
from core.import_utils import import_module
from core.process.blockwise import BlockProcessor, Var


class GenericFlags(Enum):
    """Enumeration of generic flag names"""
    LAND = "LAND"
    L1_INVALID = "L1_INVALID"
    CLOUD = "CLOUD"
    L1_DEGRADED = "L1_DEGRADED"


class FlagsReaderBase(ABC):
    """
    Abstract base class for managing flags in Earth Observation datasets.

    This class provides a standardized interface for accessing quality flags across
    different satellite sensors and data products. Each sensor may have its own flag
    naming conventions and storage formats, but this class enables uniform access
    through the GenericFlags enumeration.

    Subclasses must implement methods to:
    - Declare which dataset variables are needed for flag retrieval
    - Map generic flag names to sensor-specific flag data
    - Extract raw flags from the dataset structure

    The abstract interface ensures that all flag managers can be used interchangeably,
    allowing downstream code (in paticular, FlagsInit) to work with flags in a
    sensor-agnostic manner.
    """

    @abstractmethod
    def requires(self) -> list[str]:
        """List of variables in Dataset required by `getflag`."""
        return []

    @abstractmethod
    def dims_like(self) -> str:
        """Returns an input var name that has the same shape as the `getflag` output"""
        return ""

    @abstractmethod
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a flag from the dataset using the standard flag name.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset.
        flag_name : str or GenericFlags
            The flag name.

        Returns
        -------
        xr.DataArray
            The flag data (bool).
        """
        pass

    def getflag_raw(self, ds: xr.Dataset, flag_name: str) -> xr.DataArray:
        """
        Retrieve a raw flag from the dataset.
        
        This method is not mandatory, as it is not used by `FlagsInit`.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset.
        flag_name : str
            The raw flag name.

        Returns
        -------
        xr.DataArray
            The flag data (bool).
        """
        raise NotImplementedError


class FlagsReader(FlagsReaderBase):
    """
    A manager for per-sensor flags handling (basic implementation)

    Provides methods to retrieve flags based on a mapping from standard flag names
    to raw flag. More complex implementations can be done by subclassing the
    FlagsReaderBase class.
    """
    def __init__(self, mapping: dict[GenericFlags, str], flags_var: str):
        """
        Initialize the FlagsReader.

        Parameters
        ----------
        mapping : dict[GenericFlags, str]
            Mapping from standard flags to raw flag expressions.
            Mapping may start with a '~' for logical inversion:
            mapping = {
                GenericFlags.LAND: "~WATER",
                GenericFlags.L1_INVALID: "NODATA",
            }
        flags_var : str
            Name of the variable in the dataset containing the flags.
        """
        self.mapping = mapping
        self.flags_var = flags_var
    
    def requires(self) -> list[str]:
        """List of variables in Dataset required by `getflag`."""
        return [self.flags_var]
    
    def dims_like(self) -> str:
        """Returns an input var name that has the same shape as the output"""
        return self.flags_var
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a flag from the dataset using the standard flag name.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset.
        flag_name : str or GenericFlags
            The flag name.

        Returns
        -------
        xr.DataArray
            The flag data.
        """
        negate = False
        raw_flag_name = self.mapping[flag_name]
        if raw_flag_name.startswith('~'):
            negate = True
            raw_flag_name = raw_flag_name[1:]
        
        result = self.getflag_raw(ds, raw_flag_name)
        assert result.dtype == 'bool'
        if negate:
            return np.logical_not(result) # type: ignore
        else:
            return result

    def getflag_raw(self, ds: xr.Dataset, flag_name: str) -> xr.DataArray:
        """
        Retrieve a raw flag from the dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset.
        flag_name : str
            The raw flag name.

        Returns
        -------
        xr.DataArray
            The flag data (bool).
        """
        return tools.getflag(ds[self.flags_var], flag_name)


class FlagsInit(BlockProcessor):
    """
    Generic flags initializer that extracts quality flags from Level 1 data using a
    per-sensor flag manager.

    The flag reader instance should be stored as an attribute ('_flag_reader') in the
    Level 1 object that uses this initializer.

    Parameters
    ----------
    flags : dict
        Mapping of flag names to integer values (powers of 2 for bitmask encoding).

    dtype : str
        Data type for the flag array (typically 'uint16' for packed flags).

    flag_reader : str
        Fully qualified module path to the flag reader class (inherits from FlagsReaderBase).

    flag_reader_kwargs : dict | None, optional
        Additional keyword arguments for the flag reader constructor.

    flags_varname : str, default='flags'
        Name of the variable to create in the dataset containing the initialized flags.

    strict : bool, default=True
        If True (default), raise ValueError when a flag is not supported by the FlagsReader.
        If False, silently skip unsupported flags instead of raising an error.

    Example
    -------
    ```python
    from eoread.flags import FlagsInit, FlagsReader, GenericFlags

    # Define flag mapping (bitmask values)
    flags = {
        GenericFlags.LAND: 1 << 0,  # 1
        GenericFlags.CLOUD: 1 << 1,  # 2
    }

    # Create flags initializer
    flags_init = FlagsInit(
        flags=flags,
        dtype="uint16",
        flag_reader="eoread.flags.FlagsReader",
        flag_reader_kwargs={
            "mapping": {
                GenericFlags.LAND: "~WATER",
                GenericFlags.CLOUD: "CLOUD_MASK",
            },
            "flags_var": "quality_flags"
        }
    )
    ```
    """

    def __init__(
        self,
        flags: dict,
        dtype: str,
        flag_reader: str,
        flag_reader_kwargs: dict | None = None,
        flags_varname: str = 'flags',
        strict: bool = True,
    ):
        self.flags = flags
        self.flgreader: FlagsReaderBase = import_module(flag_reader)(**(flag_reader_kwargs or {}))
        self.dtype = dtype
        self.flags_varname = flags_varname
        self.strict = strict

    def input_vars(self) -> list[Var]:
        required = [Var(x) for x in self.flgreader.requires()]
        dimslike = Var(self.flgreader.dims_like())
        if dimslike not in required:
            required.append(dimslike)
        return required

    def created_vars(self) -> list[Var]:
        return [
            Var(
                self.flags_varname,
                dtype=self.dtype,
                flags=self.flags,
                dims_like=self.flgreader.dims_like(),
            )
        ]

    def process_block(self, block: xr.Dataset):
        flags = xr.zeros_like(block[self.flgreader.dims_like()], dtype=self.dtype)
        flags.attrs = {}
        for flagname, flag_value in self.flags.items():
            # flag_name is either a GenericFlag(enum) or a string
            try:
                flag_data = self.flgreader.getflag(block, flagname)
            except ValueError:
                if self.strict:
                    raise
                continue
            tools.raiseflag(
                flags,
                flagname,
                flag_value,
                flag_data,
            )
        block[self.flags_varname] = flags