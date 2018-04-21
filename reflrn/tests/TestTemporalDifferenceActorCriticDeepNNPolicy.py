import logging
import random
import unittest

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

from reflrn import EnvironmentLogging
from reflrn import ModelParameters
from reflrn import ReplayMemory
from reflrn import TemporalDifferenceActorCriticDeepNNPolicy
from .TestState import TestState


class TestTemporalDifferenceActorCriticDeepNNPolicy(unittest.TestCase):
    __qval_file = 'qvn_dump.pb'
    __model_file = 'model.keras'
    __lg = None

    @classmethod
    def setUpClass(cls):
        random.seed(42)
        np.random.seed(42)
        cls.__lg = EnvironmentLogging("TestTemporalDifferenceDeepNNPolicy",
                                      "TestTemporalDifferenceDeepNNPolicy.log",
                                      logging.INFO).get_logger()

    #
    # Test the NN model behaviour.
    #
    # Is the defined model able to learn and predict a single state action pair.
    #
    def test_model_behaviour_1(self):

        nn_model = TemporalDifferenceActorCriticDeepNNPolicy.create_new_model_instance()
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        y = np.array([1.1234, 2.1234, 3.1234, 4.1234, 5.1234, 6.1234, 7.1234, 8.1234, 9.1234])
        nt = 500
        xx = np.reshape(np.repeat(x, nt, axis=-1), (x.size, nt)).T
        yy = np.reshape(np.repeat(y, nt, axis=-1), (y.size, nt)).T

        err = 1e9
        iter = 0
        model_loss = 1e9
        model_acc = 1e9
        while (err > 1e-6 or model_loss > 1e-9) and model_acc > 0.75 and iter < 10:
            nn_model.fit(xx, yy, batch_size=35, epochs=50, shuffle=False)
            pr = nn_model.predict(np.array([x]))[0]
            err = np.max(np.absolute(y - pr))
            scores = nn_model.evaluate(xx, yy)
            model_loss = scores[0]
            model_acc = scores[1]
            iter += 1

        self.__lg.debug("Final Prediction: " + str(pr))
        self.assertGreater(1e-6, err)
        self.assertGreater(1e-6, model_loss)
        self.assertGreater(model_acc, 0.75)
        return

    #
    # Test the NN model behaviour.
    #
    # Is the defined model able to learn and predict 1000 state action pairs in a replay memory
    # of 10K samples with each pair repeated 10 times
    #
    def test_model_behaviour_1000(self):

        nn_model = TemporalDifferenceActorCriticDeepNNPolicy.create_new_model_instance()
        nt = 100
        x = np.empty((nt, 9))
        y = np.empty((nt, 9))
        for i in range(0, nt):
            xr = self.__rand_x()
            yr = self.__rand_y()
            x[i] = xr
            y[i] = yr

        err = 1e9
        iter = 0
        model_loss = 1e9
        model_acc = 0
        while err > 1e-6 and model_loss > 1e-9 and model_acc < 0.75 and iter < 1000:
            nn_model.fit(x, y, batch_size=33, epochs=5000, shuffle=False)
            pr = nn_model.predict(x)[0]
            err = np.max(np.absolute(y - pr))
            scores = nn_model.evaluate(x, y)
            model_loss = scores[0]
            model_acc = scores[1]
            iter += 1

        self.__lg.debug("Final Prediction: " + str(pr))
        self.assertGreater(1e-6, err)
        self.assertGreater(1e-6, model_loss)
        self.assertGreater(model_acc, 0.75)
        return

    #
    # Test the NN model behaviour.
    #
    # arbitrary quadratic
    #
    def test_model_behaviour_quad(self):

        model = Sequential()
        model.add(Dense(2, input_dim=2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        nt = 200
        x = np.empty((nt, 2))
        y = np.empty((nt, 1))
        for i in range(0, nt):
            xr = np.array([i, i * i])
            yr = 2 * xr[0] + xr[1] + 1
            x[i] = xr
            y[i] = yr

        plt.plot(x[:, 0], y)
        plt.show()

        estimator = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5, verbose=0)
        estimator.fit(x, y)
        pr = estimator.predict(x)
        err = np.max(np.absolute(y - pr))
        scores = model.evaluate(x, y)
        model_loss = scores[0]
        model_acc = scores[1]
        iter += 1

        self.__lg.debug("Final Prediction: " + str(pr))
        self.assertGreater(1e-6, err)
        self.assertGreater(1e-6, model_loss)
        self.assertGreater(model_acc, 0.75)
        return

    #
    # return a random x "state" vector
    #
    def __rand_x(self) -> np.array:
        x = np.random.randint(low=-1, high=1, size=(1, 9))
        x = x * np.float(1)
        return x

    #
    # return a random y "state" vector
    #
    def __rand_y(self) -> np.array:
        y = np.random.rand(1, 9) * 100
        return y

    #
    # Test convergence on single game pattern.
    #
    # Game is moves 0,2,4,6,8 (diagonal win)
    #
    def test_training(self):
        ts0 = TestState("000000000")
        ts1 = TestState("100000000")
        ts2 = TestState("10-1000000")
        ts3 = TestState("10-1010000")
        ts4 = TestState("10-1010-100")
        ts5 = TestState("10-1010-101")

        test_cases = (('1', ts0, ts1, 0, 0.0, False),
                      ('-1', ts1, ts2, 2, 0.0, False),
                      ('1', ts2, ts3, 4, 0.0, False),
                      ('-1', ts3, ts4, 6, 0.0, False),
                      ('1', ts4, ts5, 8, 100.0, True),
                      )

        rpmem_sz = 500
        rm = ReplayMemory(self.__lg, rpmem_sz)
        mp = ModelParameters(10, 10, 50, 10, rpmem_sz, 25)
        tdacdnnp = TemporalDifferenceActorCriticDeepNNPolicy(self.__lg, rm, "./model.test", mp, False)

        for i in range(0, 200):
            case = test_cases[i % 5]
            tdacdnnp.update_strategy(case[0], case[1], case[2], case[3], case[4], case[5])

        pa0 = tdacdnnp.select_action("1", ts0, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        pa1 = tdacdnnp.select_action("1", ts1, [0, 1, 3, 4, 5, 6, 7, 8])
        pa2 = tdacdnnp.select_action("1", ts2, [0, 1, 3, 5, 6, 7, 8])
        pa3 = tdacdnnp.select_action("1", ts3, [0, 1, 3, 5, 7, 8])
        pa4 = tdacdnnp.select_action("1", ts4, [0, 1, 3, 5, 7])

        return


#
# Execute the tests.
#


if __name__ == "__main__":
    if False:
        tests = TestTemporalDifferenceActorCriticDeepNNPolicy()
        suite = unittest.TestLoader().loadTestsFromModule(tests)
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(TestTemporalDifferenceActorCriticDeepNNPolicy("test_model_behaviour_quad"))
        unittest.TextTestRunner().run(suite)
