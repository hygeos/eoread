import pytest
import xarray as xr
import numpy as np

from eoread.flags import GenericFlags, FlagsReader, FlagsInit



def create_test_dataset():
    """Create a test dataset with quality_flags for testing FlagsReader."""
    # Create a dataset with quality_flags as integer array with bit-encoded flags
    # Bit 0 = WATER, Bit 1 = NODATA
    # 1 = WATER set, 2 = NODATA set, 3 = both set, 0 = none set
    quality_flags_data = [[1, 0], [2, 3]]  # (0,0): WATER, (0,1): none, (1,0): NODATA, (1,1): both
    
    quality_flags = xr.DataArray(
        quality_flags_data,
        dims=['x', 'y']
    )
    # Set attributes for flag meanings
    quality_flags.attrs['flag_meanings'] = 'WATER NODATA'
    quality_flags.attrs['flag_masks'] = [1, 2]
    
    ds = xr.Dataset()
    ds['quality_flags'] = quality_flags
    return ds



def test_FlagsReader():
    ds = create_test_dataset()
    
    # Create FlagsReader with mapping
    mapping = {
        GenericFlags.LAND: "~WATER",
        GenericFlags.L1_INVALID: "NODATA"
    }
    flgreader = FlagsReader(mapping, "quality_flags")
    
    # Test LAND flag (negation of WATER)
    land_result = flgreader.getflag(ds, GenericFlags.LAND)
    # WATER is bit 0, so WATER is [True, False, False, True] for positions (0,0), (0,1), (1,0), (1,1)
    # LAND = ~WATER = [False, True, True, False]
    expected_land_data = [[False, True], [True, False]]
    expected_land = xr.DataArray(expected_land_data, dims=['x', 'y'])
    xr.testing.assert_equal(land_result, expected_land)
    
    # Test L1_INVALID flag (direct NODATA = bit 1)
    invalid_result = flgreader.getflag(ds, GenericFlags.L1_INVALID)
    # NODATA is bit 1, so [False, False, True, True]
    expected_invalid_data = [[False, False], [True, True]]
    expected_invalid = xr.DataArray(expected_invalid_data, dims=['x', 'y'])
    xr.testing.assert_equal(invalid_result, expected_invalid)
    
    # Test requires
    assert flgreader.requires() == ["quality_flags"]
    
    # Test dims_like
    assert flgreader.dims_like() == "quality_flags"

def test_FlagsInit():
    # Test the flagsinit processor on a dummy input chunked xr.Dataset with
    # a variable "quality_flags" including a flag "WATER"
    ds = create_test_dataset()
    # Add chunking to make it more realistic
    ds = ds.chunk({'x': 1, 'y': 1})
    
    # Create FlagsInit processor
    # Map LAND to bit 0 (value 1), L1_INVALID to bit 2 (value 4)
    flags = {
        GenericFlags.LAND: 1,  # bit 0
        GenericFlags.L1_INVALID: 4,  # bit 2
    }
    
    # Create a FlagsReader for testing
    mapping = {
        GenericFlags.LAND: "~WATER",  # LAND is negation of WATER
        GenericFlags.L1_INVALID: "NODATA"
    }
    flag_reader_kwargs = {'mapping': mapping, 'flags_var': 'quality_flags'}
    
    flags_init = FlagsInit(
        flags=flags,
        dtype='uint8',
        flag_reader='eoread.flags.FlagsReader',
        flag_reader_kwargs=flag_reader_kwargs,
        flags_varname='flags'
    )
    
    # Process the block
    result = flags_init.map_blocks(ds)
    
    # Verify the flags were created correctly
    assert 'flags' in result
    assert result['flags'].dtype == 'uint8'
    
    # Check the flag values
    # LAND = ~WATER, so where WATER is False, LAND should be True (bit 0 set)
    # L1_INVALID = NODATA, so where NODATA is True, bit 2 should be set
    expected_flags = np.zeros((2, 2), dtype='uint8')
    # Position (0,0): WATER=True, NODATA=False -> LAND=False, L1_INVALID=False -> 0
    # Position (0,1): WATER=False, NODATA=False -> LAND=True, L1_INVALID=False -> 1
    # Position (1,0): WATER=False, NODATA=True -> LAND=True, L1_INVALID=True -> 1 + 4 = 5
    # Position (1,1): WATER=True, NODATA=True -> LAND=False, L1_INVALID=True -> 4
    expected_flags[0, 0] = 0  # no flags
    expected_flags[0, 1] = 1  # LAND
    expected_flags[1, 0] = 5  # LAND + L1_INVALID
    expected_flags[1, 1] = 4  # L1_INVALID
    
    np.testing.assert_array_equal(result['flags'].values, expected_flags)
    
    # Test input_vars and creates_vars
    input_vars = flags_init.input_vars()
    assert len(input_vars) == 1
    assert str(input_vars[0]) == 'quality_flags'
    
    created_vars = flags_init.created_vars()
    assert len(created_vars) == 1
    assert str(created_vars[0]) == 'flags'
