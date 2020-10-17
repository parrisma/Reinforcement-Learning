from typing import Tuple, List
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from examples.LSTM.UKWeatherData.UKWeatherDataLoader import UKWeatherDataLoader
from examples.LSTM.IOTA.IOTADataLoader import IOTADataLoader

"""

This is a simple LSTM test case for uni-variate time-series prediction.

Helpful Links.
    https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
    https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
"""


def data_to_look_back_data_set(data, look_back_window_size) -> np.ndarray:
    """
    Take a 1 by n vector and convert to an m by look_back_window_size array. Where the window
    is slid by one position fo reach new row of the resulting vector.
    :param data: the data set as single vector
    :param look_back_window_size:
    :return: Data in look back frames
    """
    num_frames = len(data) - (look_back_window_size - 1)
    look_back_data_set = np.zeros((num_frames, look_back_window_size))
    i = 0
    for f in range(0, num_frames):
        look_back_data_set[i] = (data[i:i + look_back_window_size]).transpose()
        i += 1
    return look_back_data_set


def build_model(look_back_window_size):
    """
    Uni-variate LSTM model with a look-back-window
    :param look_back_window_size: the size (as int) of the look back window
    :return: LSTM Model
    """
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = Sequential()
        model.add(LSTM(units=50, input_shape=(look_back_window_size, 1), return_sequences=False, name="lstm-1"))
        model.add(Dense(25, activation='relu', name="dense-1"))
        model.add(Dense(5, activation='relu', name="dense-2"))
        model.add(Dense(1, name="dense-3"))
        model.compile(loss='mse', optimizer='adam')

    print(model.summary())  # Summary to console as text

    return model


def load_weather_data(cyclic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    UKWeatherDataLoader.set_data_path(os.getcwd() + './UKWeatherData')
    _, data = UKWeatherDataLoader.load_data_set(data_set_id=UKWeatherDataLoader.DATA_SET_HEATHROW)

    y = data[:, UKWeatherDataLoader.COL_TMAX_N]

    if cyclic:
        # predict based on knowing where in year we are - path 6 months of month nums
        x = data[:, 0]
        col_id = UKWeatherDataLoader.COL_MONTH  # The month in the year 1 - 12
    else:
        # predict based on rolling prev temps.
        col_id = UKWeatherDataLoader.COL_TMAX_N  # MinMaxNormalised max temp in the month

    x = data[:, col_id]
    return x, y


def load_iota_data(cyclic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    IOTADataLoader.set_data_path(os.getcwd() + './IOTA')
    _, data = IOTADataLoader.load_data_set()

    y = data[:, IOTADataLoader.COL_CLOSE]
    x = data[:, IOTADataLoader.COL_OPEN]
    return x, y


def load_sine_wave_data() -> Tuple[np.ndarray, np.ndarray]:
    xmax = np.pi * 20.0
    x = np.arange(0, xmax, xmax / 2000)
    y = np.sin(x)
    return x, y


def split_train_test_ratio(split_ratio: float,
                           x: np.ndarray,
                           y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the test and train data in the given ratio first len(x) * ratio = Train, last len(x) * ratio = Test
    :param split_ratio: The ratio in which to split the data (
    :param x: All X
    :param y: All y
    :return: x_train, x_test, y_train, y_test
    """
    # Train with split ratio % of the data
    train_size = int(len(x) * split_ratio)

    # X is what we are predicting from
    #
    x_train = x[:train_size, :]
    x_test = x[train_size:, :]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Y is what we are trying to predict. In this case we are trying to predict tmax
    #
    y_train = y[:train_size]
    y_test = y[train_size:]
    return x_train, x_test, y_train, y_test


def split_train_test_every(every: int,
                           x: np.ndarray,
                           y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data by taking every nth element and making it test data.
    :param every: every nth element to take
    :param x: All x - len(x) must equal len(y)
    :param y: All y
    :return: x_train, x_test, y_train, y_test
    """
    shp = x.shape
    idx = np.arange(0, len(x))
    idx_2_test = slice(None, None, every)
    idx_2_train = np.delete(idx, idx_2_test)
    ltt = shp[0] - len(idx_2_train)
    ltr = shp[0] - ltt
    return np.reshape(x[idx_2_train], (ltr, shp[1], 1)), \
           np.reshape(x[idx_2_test], (ltt, shp[1], 1)), \
           np.reshape(y[idx_2_train], ltr), \
           np.reshape(y[idx_2_test], ltt)


def main():
    """
    We try 2 test cases:

    1. Case-1 => The X-Inputs are the previous tmax values and we try to learn to predict tmax given a window of
                 tmax's. In this case we should be able to learn a more sophisticated pattern - however there is
                 a risk of over fitting and not being able to generalise to the underlying yearly cycle ??

    2. Case-2 => The X-Inputs are the previous month values so all we know is where we are in an arbitrary year.
                 so we should be able to extract a cyclic pattern if there is one but at the expense of being
                 less able to predict for a specific period in time as we dont know the year or have the prev.
                 temps to drive the model. Here we learn the average values of tmax in a cycle.

    Both of these cases would work just as well predicting hours of sun, rain fall etc by picking the columns of
    data containing those values.

    :return:
    """

    # x, y = load_iota_data()
    # x, y = load_weather_data()
    x, y = load_sine_wave_data()

    # We predict based on a rolling window of 6 months
    look_back_window_size = 20

    # Convert the X data set into rolling frames e.g. [1, 2, 3, 4, 5, 6, 7] with a look back of 3
    # becomes
    # [1, 2, 3]
    # [2, 3, 4]
    # [3, 4, 5]
    # [4, 5, 6]
    # [5, 6, 7]
    # last row is partial so cannot add [6 ,7, ??]

    x = data_to_look_back_data_set(x, look_back_window_size)
    y = y[y.shape[0] - x.shape[0]:]

    # x_train, x_test, y_train, y_test = split_test_train_ratio(split_ratio=.75, x=x, y=y)
    x_train, x_test, y_train, y_test = split_train_test_every(every=4, x=x, y=y)

    # Create our simple LSTM model
    #
    lstm = build_model(look_back_window_size)

    # Quick, check to see what shapes we are passing
    # if ts = train set size
    # (ts, 6, 1)
    # (ts, )
    #
    print(x_train.shape)
    print(y_train.shape)

    # Learn ...
    #
    history = lstm.fit(x_train, y_train, epochs=500, batch_size=32, verbose=2)

    # How did we do ?
    #
    plt.plot(history.history['loss'])
    plt.show()

    # Predict first 80% based on real X
    xs = x_test.shape
    xl = xs[0]
    xr = int(xl * .99)
    y_pred = lstm.predict(tf.convert_to_tensor(x_test[:xr]))

    y_p = np.zeros((xl - xr))
    yi = 0
    xp = x_test[max(0, xr - 100):xr]  # Initial window of 10 X datums
    xp_s = xp.shape
    xp_sz = xp.size
    x_w = np.zeros((xp_sz))
    for _ in range(xr + 1, xl):
        ypt = lstm.predict(tf.convert_to_tensor(xp))
        yp = ypt[-1:].reshape(1)
        x_w[:-1] += xp.reshape(xp_sz)[1:]
        x_w[-1:] += yp
        xp = x_w.reshape((xp_s[0], xp_s[1], xp_s[2]))
        x_w = np.zeros((xp_sz))
        y_p[yi] = yp
        yi += 1
        print(str(yi))

    y_pred = np.concatenate((y_pred.reshape(y_pred.size), y_p))
    # How did we do in terms of mse ?
    err = abs(y_test - y_pred)

    # Plot Results.
    plt.title("Actual vs Predicted with mse")
    plt.plot(y_test[:-1])
    plt.plot(y_pred[:-1])
    plt.plot(err[:-1])
    plt.legend(['actual', 'predicted', 'err'], loc='upper left')
    plt.show()

    return


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    main()
