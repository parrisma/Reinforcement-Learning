import unittest
import logging
import numpy as np
from TemporalDifferencePolicy import TemporalDifferencePolicy
from State import State


class TestState(State):

    def __init__(self, state_name: str):
        self.__state_name = state_name

    def state(self) -> object:
        raise NotImplementedError("No object state of [" + self.__class__.__name__ + "] implemented")
        return None

    def state_as_string(self) -> str:
        return self.__state_name


class TestTemporalDifferencePolicy(unittest.TestCase):

        def test_1(self):
            lg = logging.getLogger(self.__class__.__name__)
            lg.addHandler(logging.NullHandler)

            test_state_1 = TestState("1")
            possible_actions = np.array([0, 1, 2])
            expected_action = 0

            tdp = TemporalDifferencePolicy(lg, "", None, False)
            tdp.update_policy("X", None, 0, test_state_1, 0, 1)
            self.assertEqual(tdp.greedy_action("TestAgentName", test_state_1, possible_actions), expected_action)

#
# Execute the tests.
#


if __name__ == "__main__":
    tests = TestTemporalDifferencePolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
