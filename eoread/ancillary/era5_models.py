from eoread.fileutils import filegen

from datetime import date
from pathlib import Path

import cdsapi

class ERA5_Models:
    
    @filegen(1)
    def reanalysis_single_level(era5, target: Path, d: date, area):
        """
        Download a single file, containing 24 times, hourly resolution
        uses the CDS API. Uses a temporary file and avoid unnecessary download 
        if it is already present, thanks to fileutil.filegen 
        
        - cams: ERA5 instance object to be passed
        - target: path to the target file after download
        - d: date of the dataset
        """
        if era5.client is None:
            era5.client = cdsapi.Client()

        print(f'Downloading {target}...')
        era5.client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': era5.cds_variables,
                'year':[f'{d.year}'],
                'month':[f'{d.month:02}'],
                'day':[f'{d.day:02}'],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', 
                         '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', 
                         '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', ],
                'format':'netcdf',
                'area': area,
            },
            target)
        return
        
    
    @filegen(1)
    def reanalysis_pressure_levels(era5, target: Path, d: date, area):
        """
        Download a single file, containing 24 times, hourly resolution
        uses the CDS API. Uses a temporary file and avoid unnecessary download 
        if it is already present, thanks to fileutil.filegen 
        
        - target: path to the target file after download
        - d: date of the dataset
        """
        if era5.client is None:
            era5.client = cdsapi.Client()

        print(f'Downloading {target}...')
        era5.client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'area': area,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'year':[f'{d.year}'],
                'month':[f'{d.month:02}'],
                'day':[f'{d.day:02}'],
                'pressure_level': [
                    '1', '2', '3',
                    '5', '7', '10',
                    '20', '30', '50',
                    '70', '100', '125',
                    '150', '175', '200',
                    '225', '250', '300',
                    '350', '400', '450',
                    '500', '550', '600',
                    '650', '700', '750',
                    '775', '800', '825',
                    '850', '875', '900',
                    '925', '950', '975',
                    '1000',
                ],
                'variable': era5.cds_variables,
            },
            target)
        return