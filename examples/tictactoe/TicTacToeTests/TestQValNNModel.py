import logging
import random
import unittest

import numpy as np

from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.QValNNModel import QValNNModel

lg = EnvironmentLogging("TestQValNNModel", "TestQValNNModel.log", logging.INFO).get_logger()


class TestQValNNModel(unittest.TestCase):
    _num_inputs = 9
    _num_actions = 9

    # Can we stand up the basic class ?
    def test_1(self):
        mdl = QValNNModel("TestModel",
                          input_dimension=TestQValNNModel._num_inputs,
                          num_actions=TestQValNNModel._num_actions,
                          lg=lg)
        self.assertIsNotNone(mdl)
        return

    # Can we create a model and compile it ?
    def test_2(self):
        mdl = QValNNModel("TestModel",
                          input_dimension=TestQValNNModel._num_inputs,
                          num_actions=TestQValNNModel._num_actions,
                          lg=lg)
        self.assertIsNotNone(mdl)
        nn = mdl.new_model()
        self.assertIsNotNone(nn)
        return

    # Can we make a (random) prediction on a complied model.
    def test_3(self):
        mdl = QValNNModel("TestModel",
                          input_dimension=TestQValNNModel._num_inputs,
                          num_actions=TestQValNNModel._num_actions,
                          lg=lg)
        try:
            mdl.new_model()
            mdl.compile()
            x = np.random.rand(1, self._num_actions)
            mdl.predict(x)
        except:
            self.assertTrue(False, "Un Expected Exception Thrown in test_3()")
        return

    # Can we teach the model a 4th order polynomial ?
    def test_4(self):
        mdl = QValNNModel("TestModel",
                          input_dimension=TestQValNNModel._num_inputs,
                          num_actions=TestQValNNModel._num_actions,
                          lg=lg)
        try:
            mdl.new_model()
            mdl.compile()
            x = np.random.rand(1, self._num_actions)
            mdl.predict(x)
        except:
            self.assertTrue(False, "Un Expected Exception Thrown in test_3()")
        return

    # return a NN X,Y training set where X is a 4th order polynomial mapped to state space Y
    def test_data(self):
        yset = dict()
        rg = np.fromiter(range(1, 10), dtype=np.float)
        num_samples = 2000
        X = np.zeros((num_samples, 1, self._num_actions))
        Y = np.zeros((num_samples, 1, self._num_inputs))
        for n in range(0, num_samples):
            while True:
                y = np.fromiter(map(lambda x: random.getrandbits(1), [None] * 9), dtype=np.float)
                ys = np.array_split(y)
                if ys not in yset:
                    yset[ys] = True
                    break
            Y[n] = y
            X[n] = np.random.rand(1, self._num_actions)
            pass


#
# Execute the Test Actor Critic Policy Unit Test Suite.
#
if __name__ == "__main__":
    tests = TestQValNNModel()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
