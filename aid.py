""" 
Aid
---
This is the Aid module.
Host of useful manipulation methods for Furgo Metocean Consultancy - Americas.
"""
from erddapy import ERDDAP
import numpy as np
import pandas as pd

class Aid():
    """ Support variables"""
    b_list = ['id', 'Label', 'Source', 'Misc', 'latitude', 'longitude', 'time']
    ERDDAP_TB = {
        'Aquadopp': {'dataset_id': 'geosOceanorAquadopp', 'vars': [
            *b_list, 'depth', 'AqSpd', 'AqDir']},
        'Adcp': {'dataset_id': 'geosOceanorADCP', 'vars': [
            *b_list, 'depth', 'CurrSpd', 'CurrDir']},
        'Wave': {'dataset_id': 'geosOceanorWave', 'vars': [
            *b_list, 'Hm0', 'Hmax', 'Hm0a', 'Hm0b', 'Tp', 'Tm02', 'Tm02a',
            'Tm02b', 'Thmax', 'Mdir', 'Mdira', 'Mdirb', 'Sprtp', 'Thtp',
            'Thlf', 'Thhf', 'Ui', 'Tm01', 'Tmm10', 'Tm24', 'Hm0_Test_Result',
            'Hmax_Test_Result', 'Hm0a_Test_Result', 'Hm0b_Test_Result']},
        'Wind': {'dataset_id': 'geosOceanorWind', 'vars': [
            *b_list, 'altitude', 'WindSpeed', 'WindDirection', 'WindGust']},
        'WindCorrected': {'dataset_id': 'geosOceanorWindCorrected10m',
                          'vars': ['id', 'longitude', 'latitude', 'time',
                                   'original_altitude', 'altitude',
                                   'WindSpeed', 'WindDirection', 'WindGust']},
        'WaterTemp': {'dataset_id': 'geosOceanorTemperature', 'vars': [
            *b_list, 'depth', 'WaterTemp']},
        'Met': {'dataset_id': 'geosOceanorMet',
                'vars': [*b_list, 'AirPressure', 'AirTemperature',
                         'AirHumidity', 'SolarRadiation']},
        'BuoyData': {'dataset_id': 'geosOceanorBuoyData', 'vars': [
            *b_list, 'LeadBatteryVoltage', 'LithBatteryVoltage',
            'AhCharged', 'AhDischargedLead', 'AhDischargedLithium',
            'LeadBatteryTemp', 'CardNo']}}             

def erddap_instance(server='http://10.1.1.17:8080/erddap',
                    protocol='tabledap', response='csv'):
    return ERDDAP(server=server, protocol=protocol, response=response)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
