import logging
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

    # Execute a single training cycle.
    def test_4(self):
        mdl = QValNNModel("TestModel",
                          input_dimension=TestQValNNModel._num_inputs,
                          num_actions=TestQValNNModel._num_actions,
                          lg=lg)
        try:
            mdl.new_model()
            mdl.compile()
            x, y = self.test_data()
            xr, yr = self.random_test_data_batch(x, y, 200)
            mdl.train(xr, yr)
            mdl.predict((xr[0]).reshape(1, TestQValNNModel._num_inputs))
        except:
            self.assertTrue(False, "Un Expected Exception Thrown in test_4()")
            raise
        return

    # Train an ensure loss reduces cycle on cycle.
    def test_5(self):
        mdl = QValNNModel("TestModel",
                          input_dimension=TestQValNNModel._num_inputs,
                          num_actions=TestQValNNModel._num_actions,
                          lg=lg)
        mdl.new_model()
        mdl.compile()
        x, y = self.test_data()
        num_tests = 10
        losses = np.zeros(num_tests)
        for i in range(0, num_tests):
            xr, yr = self.random_test_data_batch(x, y, 200)
            mdl.train(xr, yr)
            xe, ye = self.random_test_data_batch(x, y, 50)
            scores = mdl.evaluate(xe, ye)
            if type(scores) == list:
                loss = scores[0]
            else:
                loss = scores
            print("Loss: {}".format(loss))
            losses[i] = loss
        self.assertTrue(losses[0] > losses[num_tests - 1])
        return

    # return a NN X,Y training set where X is a random in range 0.0 to 1.0 (simulate a probability)
    def test_data(self):
        num_samples = (2 ** 9)
        x = np.zeros((num_samples, self._num_inputs))
        y = np.zeros((num_samples, self._num_actions))
        for n in range(0, num_samples):
            x[n] = np.asarray([int(d) for d in format(n, "0" + str(self._num_inputs) + "b")])
            y[n] = np.random.rand(9)
        return x, y

    # Select a random set of test data
    @classmethod
    def random_test_data_batch(cls, x, y, n: int):
        indices = np.random.choice(np.shape(x)[0], min(np.shape(x)[0], n), replace=False)
        xr = x[indices]
        yr = y[indices]
        return xr, yr


#
# Execute the Test Actor Critic Policy Unit Test Suite.
#
if __name__ == "__main__":
    tests = TestQValNNModel()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
