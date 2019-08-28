#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Uniform naming definition of parameters
'''

class Naming:
    '''
    Defines the generic variable names

    Pass kwargs to substitute default values with custom ones
    '''
    def __init__(self, **kwargs):
        # Radiometry
        self.Rtoa = 'Rtoa'
        self.Rtoa_desc = 'Top of Atmosphere reflectance'

        self.Ltoa = 'Ltoa'
        self.Ltoa_desc = 'Top of Atmosphere radiance'

        # Coordinates
        self.lat = 'latitude'
        self.lon = 'longitude'

        # Angles
        self.sza = 'sza'
        self.sza_desc = 'Sun zenith angle'

        self.vza = 'vza'
        self.vza_desc = 'View zenith angle'

        self.saa = 'saa'
        self.saa_desc = 'Sun azimuth angle'

        self.vaa = 'vaa'
        self.vaa_desc = 'View azimuth angle'

        # Attributes
        self.datetime = 'datetime'

        # Dimensions
        self.rows = 'rows'
        self.columns = 'columns'
        self.dim2 = (self.rows, self.columns)

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

