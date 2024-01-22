from ..utils.fileutils import filegen

from datetime import date
from pathlib import Path

import cdsapi

class CAMS_Models:
        
    @filegen(1)
    def global_atmospheric_composition_forecast(cams, target: Path, d: date, area):
        """
        Download a single file, containing 24 times, hourly resolution
        uses the CDS API. Uses a temporary file and avoid unnecessary download 
        if it is already present, thanks to fileutil.filegen 
        
            model example:   
        https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts
        https://confluence.ecmwf.int/display/CKB/CAMS%3A+Global+atmospheric+composition+forecast+data+documentation#heading-Table1SinglelevelFastaccessparameterslastreviewedon02Aug2023
        
        
        - cams: CAMS instance object to be passed
        - target: path to the target file after download
        - d: date of the dataset
        """
        
        if area is None:
            area = [90, -180, -90, 180]
        
        if cams.client is None:
            cams.client = cdsapi.Client(url=cams.cdsapi_cfg['url'], 
                                        key=cams.cdsapi_cfg['key'])
            
        cams.client.retrieve(
            'cams-global-atmospheric-composition-forecasts',
            {
                'date': str(d)+'/'+str(d),
                'type': 'forecast',
                'format': 'netcdf',
                'variable': cams.ads_variables,
                'time': ['00:00', '12:00'],
                'leadtime_hour': ['0', '1', '2', '3', '4', '5', 
                                  '6', '7', '8', '9', '10', '11'],
                'area': area,
            }, target)