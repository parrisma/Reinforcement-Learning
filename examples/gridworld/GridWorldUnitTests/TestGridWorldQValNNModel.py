import logging
import random
import unittest

import numpy as np
import tensorflow as tf

from examples.gridworld.TestRigs.GridWorldQValNNModel import GridWorldQValNNModel
from reflrn.TemporalDifferenceQValPolicyPersistance import TemporalDifferenceQValPolicyPersistance
from reflrn.EnvironmentLogging import EnvironmentLogging


class TestGridWorldQValNNModel(unittest.TestCase):
    __lg = None

    #
    # Load csv (
    #
    # Test level bootstrap of common elements
    #
    def setUp(self):
        self.__lg = EnvironmentLogging(self.__class__.__name__, self.__class__.__name__ + ".log",
                                       logging.DEBUG).get_logger()
        self.__tdqvpp = TemporalDifferenceQValPolicyPersistance()
        np.random.seed(42)
        tf.set_random_seed(42)

    #
    # See if the model can learn a saved set of stable QValues from a 10 by 10 grid
    # with max 4 actions per grid cell (N,S,E,W)
    #
    def test_model_on_10_by_10_saved_stable_qvals(self):

        qv = self.__load_saved_qvals(filename="TenByTenGridOfQValsTestCase1.pb",
                                     num_actions=4)

        model = GridWorldQValNNModel(model_name="TestModel1",
                                     input_dimension=2,
                                     num_actions=4,
                                     num_grid_cells=(10 * 10),
                                     lg=self.__lg,
                                     batch_size=1,
                                     num_epoch=1,
                                     lr_0=0.005,
                                     lr_min=0.001
                                     )

        xa, ya = self.__get_all(qv)  # self._get_sample_batch(qv, 10, 4)

        num_cycles = 5
        num_epoch = 8000
        report_every = 10
        for z in range(0, num_cycles):
            model.new_model()
            model.compile()
            cst = np.zeros(int(num_epoch / report_every))
            model.reset()
            j = 0
            for i in range(0, num_epoch):
                x, y = self.__get_sample_batch(qv, 32, 4)
                model.train(x=x, y=y)
                if i % report_every == 0:
                    yp = model.predict(xa)
                    cst[j] += np.sum(np.power(ya[0] - yp[0], 2))
                    self.__lg.debug(str(i) + "," + str(cst[j]))
                    j += 1
            yp = model.predict(xa)
            for i in range(1, len(ya)):
                self.__lg.debug("=")
                self.__lg.debug(np.sum(np.power(ya[i] - yp[i], 2)))
                self.__lg.debug(str(ya[i]) + ' : ' + str(np.max(ya[i])))
                self.__lg.debug(str(yp[i]) + ' : ' + str(np.max(yp[i])))
                self.__lg.debug("=")
            j = 0
        cst[j] /= num_cycles
        for i in range(0, len(cst)):
            self.__lg.debug(cst[i])

        return

    #
    # Load the set of Q Values as saved out in a test run from TemporalDifferenceQValPolicyPersistance
    # this is a sparse by curr_coords and action, so we need to convert to dense by action as we are going
    # to teach the NN to learn QVal by curr_coords and we need the Qval as dense Y input to match a given
    # X input curr_coords.
    #
    def __load_saved_qvals(self,
                           filename: str,
                           num_actions: int
                           ) -> dict:
        # Load Q Value Dictionary. This is stored sparse by curr_coords an action.
        (qv,
         n,
         learning_rate_0,
         discount_factor,
         learning_rate_decay) = self.__tdqvpp.load(filename)

        # Convert to dictionary of list, where list is dense list by action.
        qvl = dict()
        for k1 in list(qv.keys()):
            lqv = np.full(num_actions, np.float(0))  # np.inf)
            for k2 in list(qv[k1].keys()):
                lqv[int(k2)] = qv[k1][k2]
            qvl[k1] = lqv
        return qvl

    #
    # Get a random set of samples from the given QValues to select_action as a training
    # batch for the model.
    #
    @classmethod
    def __get_sample_batch(cls,
                           qvals: dict,
                           batch_size: int,
                           num_actions: int):
        x = np.zeros((batch_size, 2))
        y = np.zeros((batch_size, num_actions))
        kys = list(qvals.keys())
        for i in range(0, batch_size):
            k1 = random.choice(kys)
            xn = np.zeros(2)
            xn[0], xn[1] = k1.split(",")
            x[i] = xn
            y[i] = qvals[k1]

        return x, y

    #
    # Get a random set of samples from the given QValues to select_action as a training
    # batch for the model.
    #
    @classmethod
    def __get_all(cls,
                  qvals: dict):
        kys = list(qvals.keys())
        x = np.zeros((len(kys), 2))
        y = np.zeros((len(kys), 4))
        kys = list(qvals.keys())
        i = 0
        for k1 in kys:
            xn = np.zeros(2)
            xn[0], xn[1] = k1.split(",")
            x[i] = xn
            y[i] = qvals[k1]
            i += 1

        return x, y


#
# Execute the Test Suite.
#
if __name__ == "__main__":
    tests = TestGridWorldQValNNModel()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
