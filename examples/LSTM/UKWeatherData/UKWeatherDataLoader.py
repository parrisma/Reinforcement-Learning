"""
The data is from UK Met Office
    https://www.metoffice.gov.uk/public/weather/climate-historic/#?tab=climateHistoric

Some simple clean-up and normalization is performed on the data and it is then saved by location
into a CSV file with the following headings

location    - location id (made up ~ kept unique for all locations in the dir UK-Weather-Data
month       - Month number 1 - 12
year        - year yyyy
tmax        - max temperature in the month
tmin        - min temperature in the month
frost days  - number of days where there was a frost in the month
rain mm     - mm or rain in the month
sun hours   - number of hours of sun in the month
tmax-n      - feature scaled tmax
tmin-n      - feature scaled tmin
frost-n     - feature scaled frost days
rain-n      - feature scaled rain
sun-n       - feature scaled sun
"""

import csv

import numpy as np


class UKWeatherDataLoader:
    COL_LOCATION = 0
    COL_MONTH = COL_LOCATION + 1
    COL_YEAR = COL_MONTH + 1
    COL_TMAX = COL_YEAR + 1
    COL_TMIN = COL_TMAX + 1
    COL_FROST_DAYS = COL_TMIN + 1
    COL_RAIN_MM = COL_FROST_DAYS + 1
    COL_SUN_HRS = COL_RAIN_MM + 1
    COL_TMAX_N = COL_SUN_HRS + 1
    COL_TMIN_N = COL_TMAX + 1
    COL_FROST_DAYS_N = COL_TMIN + 1
    COL_RAIN_MM_N = COL_FROST_DAYS + 1
    COL_SUN_HRS_N = COL_RAIN_MM + 1

    DATA_SET_HEATHROW = 0
    DATA_SET_LERWICK = 1
    DATA_SET_CAMBORN = 2

    DATA_SET_HEATHROW_NAME = 'HeathrowStation'
    DATA_SET_LERWICK_NAME = ''
    DATA_SET_CAMBORN_NAME = ''

    DATA_SET = [DATA_SET_HEATHROW_NAME,
                DATA_SET_LERWICK_NAME,
                DATA_SET_CAMBORN_NAME
                ]

    path_to_data = None

    def __init__(self):
        return

    @classmethod
    def set_data_path(cls,
                      path_to_data):
        """
        Set the path to the folder that holds the data files.
        :param path_to_data:
        """
        UKWeatherDataLoader.path_to_data = path_to_data
        return

    @classmethod
    def load_data_set(cls,
                      data_set_id):
        """
        Load a data set from cvs file. The data and csv are specific to this simple test rig.
        All data columns are known to be numeric and are converted to numeric when loaded into the
        numpy array.

        :return: numpy array holding loaded data
        """
        with open(UKWeatherDataLoader.path_to_data + '/' + cls.DATA_SET[data_set_id] + '.csv', newline='') as datafile:
            data_as_list = list(csv.reader(datafile))

        data_headers = data_as_list[0]
        del data_as_list[0]
        data_as_np = np.asarray(data_as_list)
        data_as_np = data_as_np.astype(np.float)
        return data_headers, data_as_np
