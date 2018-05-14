import logging
import unittest

from reflrn.TemporalDifferenceQValPolicyPersistance import TemporalDifferenceQValPolicyPersistance


class TestGridWorldQValNNModel(unittest.TestCase):
    __lg = None

    #
    # Load csv (
    #
    # Test level bootstrap of common elements
    #
    def setUp(self):
        self.__tdqvpp = TemporalDifferenceQValPolicyPersistance()

    #
    # See if the model can learn a saved set of stable QValues from a 10 by 10 grid
    # with max 4 actions per grid cell (N,S,E,W)
    #
    def test_model_on_10_by_10_saved_stable_qvals(self):
        qv, n = self.__load_saved_qvals("TenByTenGridOfQValsTestCase1.pb")
        for k in list(qv.keys()):
            print(qv[k])
        print(n)
        pass

    def __load_saved_qvals(self, filename: str) -> dict:
        (q_values,
         n,
         learning_rate_0,
         discount_factor,
         learning_rate_decay) = self.__tdqvpp.load(filename)

        return q_values, n


#
# Execute the Test Suite.
#
if __name__ == "__main__":
    tests = TestGridWorldQValNNModel()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
