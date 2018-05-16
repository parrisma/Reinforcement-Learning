import random
import unittest
import logging
import numpy as np

from reflrn.TemporalDifferenceQValPolicyPersistance import TemporalDifferenceQValPolicyPersistance
from examples.gridworld.TestRigs.GridWorldQValNNModel import GridWorldQValNNModel


class TestGridWorldQValNNModel(unittest.TestCase):
    __lg = None

    #
    # Load csv (
    #
    # Test level bootstrap of common elements
    #
    def setUp(self):
        self.__lg = logging.getLogger(self.__class__.__name__)
        self.__tdqvpp = TemporalDifferenceQValPolicyPersistance()

    #
    # See if the model can learn a saved set of stable QValues from a 10 by 10 grid
    # with max 4 actions per grid cell (N,S,E,W)
    #
    def test_model_on_10_by_10_saved_stable_qvals(self):

        qv = self.__load_saved_qvals(filename="TenByTenGridOfQValsTestCase1.pb",
                                     num_actions=4)

        x, y = self.__get_all(qv)  # self.__get_sample_batch(qv, 10, 4)

        model = GridWorldQValNNModel(model_name="TestModel1",
                                     input_dimension=2,
                                     num_actions=4,
                                     num_grid_cells=(10 * 10),
                                     lg=self.__lg,
                                     batch_size=1,
                                     num_epoch=1
                                     )

        ml = model.new_model()
        model.compile()
        ml.fit(x=x, y=y, validation_split=0.2, batch_size=32, nb_epoch=50000, verbose=2)
        # for i in range(0, 100000):
        #    x, y = self.__get_sample_batch(qv, 10, 4)
        #    model.train(x, y)
        x, y = self.__get_sample_batch(qv, 10, 4)
        xx = np.zeros((1, 2))
        xx[0] = x[0]
        yp = model.predict(xx)
        print(y[0] - yp[0])

        return

    #
    # Load the set of Q Values as saved out in a test run from TemporalDifferenceQValPolicyPersistance
    # this is a sparse by state and action, so we need to convert to dense by action as we are going
    # to teach the NN to learn QVal by state and we need the Qval as dense Y input to match a given
    # X input state.
    #
    def __load_saved_qvals(self,
                           filename: str,
                           num_actions: int
                           ) -> dict:
        # Load Q Value Dictionary. This is stored sparse by state an action.
        (qv,
         n,
         learning_rate_0,
         discount_factor,
         learning_rate_decay) = self.__tdqvpp.load(filename)

        # Convert to dictionary of list, where list is dense list by action.
        qvl = dict()
        for k1 in list(qv.keys()):
            lqv = np.full(num_actions, np.float(-100))  # np.inf)
            for k2 in list(qv[k1].keys()):
                lqv[int(k2)] = qv[k1][k2]
            qvl[k1] = lqv
        return qvl

    #
    # Get a random set of samples from the given QValues to act as a training
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
    # Get a random set of samples from the given QValues to act as a training
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
