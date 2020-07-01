"""
All time daily data as at June 2020 for IOTA
date: yyyy-mm-dd hh:mm:ss
weighted
close
high
low
open
volume
"""

import datetime as datetime
import csv
import numpy as np


class IOTADataLoader:
    COL_DATE = 0
    COL_WEIGHT = COL_DATE + 1
    COL_CLOSE = COL_WEIGHT + 1
    COL_HIGH = COL_CLOSE + 1
    COL_LOW = COL_HIGH + 1
    COL_OPEN = COL_LOW + 1
    COL_VOLUME = COL_OPEN + 1

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
        IOTADataLoader.path_to_data = path_to_data
        return

    @classmethod
    def load_data_set(cls):
        """
        Load a data set from cvs file. The data and csv are specific to this simple test rig.
        All data columns are known to be numeric and are converted to numeric when loaded into the
        numpy array.

        :return: numpy array holding loaded data
        """
        with open(IOTADataLoader.path_to_data + '/' + 'IOTA-DailyPriceDataAllTime' + '.csv', newline='') as datafile:
            data_as_list = list(csv.reader(datafile))

        data_headers = data_as_list[0]
        del data_as_list[0]
        for r in data_as_list:
            try:
                r[0] = datetime.datetime.strptime(r[0][:10], '%Y-%m-%d').timestamp()
            except Exception as e:
                print(str(e))
        data_as_np = np.asarray(data_as_list)
        data_as_np = data_as_np.astype(np.float)
        return data_headers, data_as_np
