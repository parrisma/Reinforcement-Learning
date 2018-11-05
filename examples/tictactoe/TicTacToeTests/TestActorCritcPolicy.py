import logging
import unittest

import numpy as np

from examples.tictactoe.TicTacToe import TicTacToe
from reflrn.ActorCriticPolicy import ActorCriticPolicy
from reflrn.EnvironmentLogging import EnvironmentLogging
from .TestAgent import TestAgent

lg = EnvironmentLogging("TestActorCriticPolicy", "TestActorCriticPolicy.log", logging.INFO).get_logger()


class TestActorCriticPolicy(unittest.TestCase):

    def test_1(self):
        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        ttt = TicTacToe(agent_x, agent_o, None)
        acp = ActorCriticPolicy(ttt, lg)
        self.assertIsNotNone(acp)

    def test_2(self):
        dt = self.get_data()
        return

    #
    # Create 20 random state to action mappings and then create a training set of 2000
    # by extracting samples at random (with repeats) from the 20 state action mappings.
    #
    def get_data(self) -> dict:
        num_samples = 20
        samples = dict()
        np.random.seed(42)  # Thanks Douglas.
        while len(samples) < num_samples:
            n = np.random.randint(0, (2 ** 9) - 1)
            if n not in samples:
                samples[n] = np.binary_repr(n, width=9)

        dt = dict()
        skeys = list(samples.keys())
        for i in range(0, 2000):
            dt[i] = [samples[skeys[np.random.randint(0, 19)]], np.random.randint(0, 9)]

        return dt


#
# Execute the Test Actor Critic Policy Unit Test Suite.
#
if __name__ == "__main__":
    tests = TestActorCriticPolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
