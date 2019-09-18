#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for exploiting eoread objects
'''

from dateutil.parser import parse

def datetime(ds):
    '''
    Parse datetime (in isoformat) from `ds` attributes
    '''
    return parse(ds.datetime)
