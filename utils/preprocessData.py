from typing import Tuple, Any

import pandas as pd
from pandas import DataFrame

from utils.addRemainingUsefulLife import add_rul


def load(path: str) -> Tuple[DataFrame | Any, DataFrame | Any, DataFrame | Any, DataFrame | Any]:
    # define filepath to read data
    dir_path = path

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)

    _y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    train = add_rul(train)
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = index_names + setting_names + drop_sensors

    _x_train = train.drop(drop_labels, axis=1)
    _y_train = _x_train.pop('RUL')

    # Since the true RUL values for the test set are only provided for the last time cycle of each engine,
    # the test set is subset to represent the same
    _x_test = test.groupby('unit_nr').last().reset_index().drop(drop_labels, axis=1)

    _x_train.to_csv(dir_path + 'X_train_FD001.csv', header=None)
    _y_train.to_csv(dir_path + 'y_train_FD001.csv', header=None)
    _x_test.to_csv(dir_path + 'X_test_FD001.csv', header=None)

    _x_train = pd.read_csv(dir_path + 'X_train_FD001.csv', index_col=0, header=None)
    _y_train_ = pd.read_csv(dir_path + 'y_train_FD001.csv', index_col=0, header=None)
    _x_test = pd.read_csv(dir_path + 'X_test_FD001.csv', index_col=0, header=None)

    return _x_train, _y_train_, _x_test, _y_test
