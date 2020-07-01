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


def data_to_look_back_data_set(data, look_back_window_size):
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


def load_weather_data(case: int = 1) -> Tuple[List, List]:
    UKWeatherDataLoader.set_data_path(os.getcwd() + './UKWeatherData')
    _, data = UKWeatherDataLoader.load_data_set(data_set_id=UKWeatherDataLoader.DATA_SET_HEATHROW)

    y = data[:, UKWeatherDataLoader.COL_TMAX_N]

    if case == 1:
        # predict based on rolling prev temps.
        col_id = UKWeatherDataLoader.COL_TMAX_N  # MinMaxNormalised max temp in the month
    else:
        # predict based on knowing where in year we are - path 6 months of month nums
        x = data[:, 0]
        col_id = UKWeatherDataLoader.COL_MONTH  # The month in the year 1 - 12
    x = data[:, col_id]
    return x, y


def load_iota_data() -> Tuple[List, List]:
    IOTADataLoader.set_data_path(os.getcwd() + './IOTA')
    _, data = IOTADataLoader.load_data_set()

    y = data[:, IOTADataLoader.COL_CLOSE]
    x = data[:, IOTADataLoader.COL_OPEN]
    return x, y


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

    x, y = load_iota_data()

    # We predict based on a rolling window of 6 months
    look_back_window_size = 6

    # Convert the X data set into rolling frames e.g. [1, 2, 3, 4, 5, 6, 7] with a look back of 3
    # becomes
    # [1, 2, 3]
    # [2, 3, 4]
    # [3, 4, 5]
    # [4, 5, 6]
    # [5, 6, 7]
    # last row is partial so cannot add [6 ,7, ??]

    x = data_to_look_back_data_set(x, look_back_window_size)

    # Train with 70% of the data
    train_size = int(len(x) * 0.7)

    # X is what we are predicting from
    #
    x_train = x[:train_size, :]
    x_test = x[train_size:, :]

    # Shape needed for LSTM input is [None, look_back, num_prediction_vars]
    # Where
    #    None           = Batch Size - such that it can vary at run time
    #    look_back      = how many rolling observations we are predicting from
    #    num_pred_vars  = how many variables are we predicting from - this is uni-variate test
    #                   so 1 = either tmax *or* month num : as per test case 1. and 2.

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Y is what we are trying to predict. In this case we are trying to predict tmax
    #
    y_train = y[:train_size]
    y_test = y[train_size:]

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
    history = lstm.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2)

    # How did we do ?
    #
    plt.plot(history.history['loss'])
    plt.show()

    # Predict first 80% based on real X
    xs = x_test.shape
    xl = xs[0]
    xr = int(xl * .8)
    y_pred = lstm.predict(x_test[:xr])

    y_p = np.zeros((xl - xr))
    yi = 0
    xp = x_test[xr:xr + 10]  # Initial window of 10 X datums
    x_w = np.zeros((xp.size))
    for _ in range(xr + 1, xl):
        yp = (lstm.predict(tf.convert_to_tensor(xp)))[-1:].reshape(1)
        x_w[:-1] += xp.reshape(60)[1:]
        x_w[-1:] += yp
        xp = x_w.reshape((10, 6, 1))
        x_w = np.zeros((xp.size))
        y_p[yi] = yp
        yi += 1
        print(str(yi))

    # How did we do in terms of mse ?
    mse_1 = (np.square(y_test - y_pred)).mean(axis=0)

    # Plot Results.
    plt.title("Actual vs Predicted with mse")
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.plot(mse_1)
    plt.legend(['actual', 'predicted', 'diff', 'mse'], loc='upper left')
    plt.show()

    return


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    main()
