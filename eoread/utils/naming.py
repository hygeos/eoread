#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Uniform naming definition of parameters
'''

from pathlib import Path

########################################################################################
# THIS MODULE WILL BE DEPRECATED, PLEASE DO NOT UPDATE THIS CODE AS IT WILL BE DELETED #
########################################################################################


class Naming:
    '''
    Defines the generic variable names

    Pass kwargs to substitute default values with custom ones
    '''
    def __init__(self, **kwargs):
        # Radiometry
        self.Rtoa      = 'Rtoa'
        self.Rtoa_desc = 'Top of Atmosphere reflectance'

        self.Ltoa      = 'Ltoa'
        self.Ltoa_desc = 'Top of Atmosphere radiance'

        self.Ltoa_tir      = 'Ltoa_tir'
        self.Ltoa_tir_desc = 'Top of Atmosphere radiance for thermal bands'

        self.BT      = 'BT'
        self.BT_desc = 'Brightness Temperature'

        self.Rw      = 'Rw'
        self.Rw_desc = 'Water reflectance'

        self.wav      = 'wav'
        self.wav_desc = 'Effective wavelength'

        self.wav_tir      = 'wav_tir'
        self.wav_tir_desc = 'Effective wavelength for thermal bands'

        self.cwav      = 'cwav'
        self.cwav_desc = 'Central (nominal) wavelength'

        self.F0      = 'F0'
        self.F0_desc = 'Solar irradiance'

        # Coordinates
        self.lat = 'latitude'
        self.lon = 'longitude'

        # Angles
        self.sza      = 'sza'
        self.sza_desc = 'Sun zenith angle'

        self.vza      = 'vza'
        self.vza_desc = 'View zenith angle'

        self.saa      = 'saa'
        self.saa_desc = 'Sun azimuth angle'

        self.vaa      = 'vaa'
        self.vaa_desc = 'View azimuth angle'

        self.raa      = 'raa'
        self.raa_desc = 'Relative azimuth angle'

        # Flags
        self.flags          = 'flags'
        self.flags_dtype    = 'uint16'
        self.flags_meanings = 'flag_meanings'
        self.flags_masks    = 'flag_masks'
        self.flags_meanings_separator = ' '

        # Attributes
        self.crs         = 'crs'
        self.datetime    = 'datetime'
        self.totalheight = 'totalheight'
        self.totalwidth  = 'totalwidth'
        self.platform    = 'platform'
        self.sensor      = 'sensor'
        self.shortname   = 'shortname'
        self.resolution  = 'resolution'
        self.description = 'description'
        self.footprint_lat = 'footprint_lat'
        self.footprint_lon = 'footprint_lon'
        self.product_name  = 'product_name'
        self.input_directory = 'input_directory'

        # Dimensions
        self.bands_tir = 'bands_tir'
        self.bands   = 'bands'
        self.rows    = 'y'
        self.columns = 'x'
        self.dim2 = (self.rows, self.columns)
        self.dim3 = (self.bands, self.rows, self.columns)
        self.dim3_tir = (self.bands_tir, self.rows, self.columns)

        # Specific dimensions
        self.bands_tir = 'bands_tir'

        # Ancillary data
        self.total_column_ozone = 'total_column_ozone'
        self.sea_level_pressure = 'sea_level_pressure'
        self.total_columnar_water_vapour = 'total_columnar_water_vapour'
        self.horizontal_wind = 'horizontal_wind'

        # dtype for floats
        self.expected_dtypes = {
            self.Rtoa : 'float32',
            self.lat  : 'float32',
            self.lon  : 'float32',
            self.vza  : 'float32',
            self.sza  : 'float32',
            self.raa  : 'float32',
            self.flags: 'uint16',
        }

        for k, v in kwargs.items():
            assert k in self.__dict__
            self.__dict__[k] = v


    def name(self, name):
        assert name in self.__dict__
        return self.__dict__[name]


    def desc(self, name):
        assert name in self.__dict__
        assert name+'_desc' in self.__dict__
        return self.__dict__[name+'_desc']

naming = Naming()

flags = {
    'LAND'          : 1,
    'CLOUD_BASE'    : 2,
    'L1_INVALID'    : 4,
}
