import unittest
import logging
import numpy as np
import os.path
from TemporalDifferencePolicy import TemporalDifferencePolicy
from TemporalDifferencePolicyPersistance import TemporalDifferencePolicyPersistance
from State import State
from random import randint


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

        # Generate a temp file name, used so we can save/load/test the resulting q values
        #
        def temp_file_name(self) ->str:
            while True:
                fn = self.__temp_file_dir+"/"+str(randint(0, 999999999))+".tmp"
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
            reward=1.0

            tdp = TemporalDifferencePolicy(lg=self.__lg, filename="", fixed_games=None, load_file=False)
            tdp.update_policy(agent_name=test_agent_1,
                              prev_state=None,
                              prev_action=0,
                              state=test_state_1,
                              action=expected_action,
                              reward=reward)
            tdp.save(temp_file)

            # Only 1 action, and that action has a reward so must be the greedy action
            self.assertEqual(tdp.greedy_action("TestAgentName", test_state_1, possible_actions), expected_action)

            (qv_dict, n, lr0, df, lrd) = tdp.load(temp_file)

            lr = self.lr(n, lr0, lrd)
            qv = 0
            qv = (1 - lr)*qv + (lr*reward)
            self.assertEqual(qv_dict[test_state_1.state_as_string()][expected_action], qv)

            return

#
# Execute the tests.
#


if __name__ == "__main__":
    tests = TestTemporalDifferencePolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
