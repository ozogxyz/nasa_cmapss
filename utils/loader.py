from typing import Any

import pandas as pd
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from sklearn.preprocessing import MinMaxScaler


def load_FD001() -> tuple[DataFrameGroupBy, DataFrameGroupBy, DataFrame | Any]:
    """

    param bound: upper limit for target RULs
    :return: grouped data per sample
    """
    # define filepath to read data
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features, derived from EDA
    drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors

    train.drop(labels=drop_labels, axis=1, inplace=True)
    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]
    data_normalized = MinMaxScaler(data)
    train_normalized = pd.concat([title, data_normalized], axis=1)
    train_group = train_normalized.groupby(by="unit_nr")

    test.drop(labels=drop_labels, axis=1, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]
    data_normalized = MinMaxScaler(data)
    test_normalized = pd.concat([title, data_normalized], axis=1)
    test_group = test_normalized.groupby(by="unit+nr")
    return train_group, test_group, y_test
