import logging
import unittest

import numpy as np

from examples.tictactoe.TicTacToe import TicTacToe
from examples.tictactoe.TicTacToeTests.TestState import TestState
from reflrn.ActorCriticPolicy import ActorCriticPolicy
from reflrn.EnvironmentLogging import EnvironmentLogging
from .TestAgent import TestAgent

lg = EnvironmentLogging("TestActorCriticPolicy", "TestActorCriticPolicy.log", logging.INFO).get_logger()


class TestActorCriticPolicy(unittest.TestCase):

    def test_simple_bootstrap(self):
        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        ttt = TicTacToe(agent_x, agent_o, None)
        acp = ActorCriticPolicy(ttt, lg)
        self.assertIsNotNone(acp)

    def test_2(self):
        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        ttt = TicTacToe(agent_x, agent_o, None)
        acp = ActorCriticPolicy(ttt, lg)

        ns = None
        dt = self.get_data()
        ag = agent_o
        i = 0
        el = np.random.randint(5, 9)
        for k in dt.keys():
            st = TestState((dt[k])[0])
            a = (dt[k])[1]
            if ns is None:
                ns = st
            else:

                acp.update_policy(agent_name=ag.name(),
                                  state=st,
                                  next_state=ns,
                                  action=a,
                                  reward=float(0),
                                  episode_complete=(i == el))
                if ag == agent_o:
                    ag = agent_x
                else:
                    ag = agent_o

                ns = st
                if i >= el:
                    i = 0
                    el = np.random.randint(5, 9)
                else:
                    i += 1

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
                samples[n] = [self.binary_as_one_hot(np.binary_repr(n, width=9)), np.random.randint(0, 9)]

        dt = dict()
        skeys = list(samples.keys())
        for i in range(0, 2000):
            ky = skeys[np.random.randint(0, 19)]
            dt[i] = samples[ky]

        return dt

    #
    # Binary string to numpy array one hot encoding as float
    #
    @classmethod
    def binary_as_one_hot(cls,
                          bstr: str) -> np.array:
        npa = np.array(list(map(float, bstr)))
        return npa


#
# Execute the Test Actor Critic Policy Unit Test Suite.
#
if __name__ == "__main__":
    tests = TestActorCriticPolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
