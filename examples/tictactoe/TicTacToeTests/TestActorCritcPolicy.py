import logging
import unittest

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


#
# Execute the Test Actor Critic Policy Unit Test Suite.
#
if __name__ == "__main__":
    tests = TestActorCriticPolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
