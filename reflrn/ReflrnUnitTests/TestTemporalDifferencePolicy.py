import logging
import os.path
import unittest
from random import randint

import numpy as np

from reflrn.Interface.State import State
from reflrn.Interface.TemporalDifferencePolicyPersistance import TemporalDifferencePolicyPersistance
from reflrn.TemporalDifferenceQValPolicy import TemporalDifferenceQValPolicy


class TestState(State):

    def __init__(self, state_name: str):
        self.__state_name = state_name

    def state(self) -> object:
        raise NotImplementedError("No object state of [" + self.__class__.__name__ + "] implemented")
        return None

    def state_as_string(self) -> str:
        return self.__state_name


class TestTemporalDifferencePolicy(unittest.TestCase):
    __lg = None
    __temp_file_dir = "./temp"
    __tddp = None

    # Generate a temp file name, used so we can save/load/test the resulting q values
    #
    def temp_file_name(self) -> str:
        while True:
            fn = self.__temp_file_dir + "/" + str(randint(0, 999999999)) + ".tmp"
            if not os.path.isfile(fn):
                break
        return fn

    # Test level bootstrap of common elements
    #
    def setUp(self):
        self.__lg = logging.getLogger(self.__class__.__name__)
        self.__lg.addHandler(logging.NullHandler)
        self.__tdpp = TemporalDifferencePolicyPersistance()

    # learning rate (alpha)
    @classmethod
    def lr(cls, n: int, lr0: float, lrd: float):
        return lr0 / (1 + (n * lrd))

    # Test of a single update to a single state.
    # V{k+1}(S) <- (1 - lr)*V{k}(S) + lr*reward + lr*df*V{k}(S')
    #
    def test_single_action_state_update(self):

        temp_file = self.temp_file_name()

        test_state_1 = TestState("1")
        test_agent_1 = "A"
        possible_actions = np.array([0, 1, 2])
        expected_action = 0
        reward = 1.0

        tdp = TemporalDifferenceQValPolicy(lg=self.__lg, filename="", fixed_games=None, load_qval_file=False)
        tdp.update_strategy(agent_name=test_agent_1,
                            prev_state=None,
                            prev_action=0,
                            state=test_state_1,
                            action=expected_action,
                            reward=reward)
        tdp.save(temp_file)

        # Only 1 action, and that action has a reward so must be the greedy action
        self.assertEqual(tdp.select_action("TestAgentName", test_state_1, possible_actions), expected_action)

        (qv_dict, n, lr0, df, lrd) = tdp.load(temp_file)
        os.remove(temp_file)

        lr = self.lr(n, lr0, lrd)
        qv = 0
        qv = (1 - lr) * qv + (lr * reward)
        self.assertAlmostEqual(qv_dict[test_state_1.state_as_string()][expected_action], qv, 9)
        return

    # Test of a 100 update to a single state, should approach the reward value to within
    # a small margin ~ 0.01 diff.
    # V{k+1}(S) <- (1 - lr)*V{k}(S) + lr*reward + lr*df*V{k}(S')
    #
    def test_100_action_state_update(self):

        temp_file = self.temp_file_name()

        test_state_2 = TestState("2")
        test_agent_1 = "A"
        possible_actions = np.array([0, 1, 2])
        expected_action = 0
        reward = 1.0
        niter = 100

        tdp = TemporalDifferenceQValPolicy(lg=self.__lg, filename="", fixed_games=None, load_qval_file=False)
        for i in range(0, niter):
            tdp.update_strategy(agent_name=test_agent_1,
                                prev_state=None,
                                prev_action=0,
                                state=test_state_2,
                                action=expected_action,
                                reward=reward)
        tdp.save(temp_file)

        # Only 1 action, and that action has a reward so must be the greedy action
        self.assertEqual(tdp.select_action("TestAgentName", test_state_2, possible_actions), expected_action)

        (qv_dict, n, lr0, df, lrd) = tdp.load(temp_file)
        os.remove(temp_file)

        qv = 0
        for i in range(0, niter):
            lr = self.lr(i, lr0, lrd)
            qv = (1 - lr) * qv + (lr * reward)
        self.assertAlmostEqual(qv_dict[test_state_2.state_as_string()][expected_action], qv, 9)
        return


#
# Execute the ReflrnUnitTests.
#


if __name__ == "__main__":
    tests = TestTemporalDifferencePolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
