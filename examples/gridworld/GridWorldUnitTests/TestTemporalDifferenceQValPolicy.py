import logging
import unittest

from reflrn.TemporalDifferenceQValPolicyPersistance import TemporalDifferenceQValPolicyPersistance
from reflrn.TemporalDifferenceQValPolicy import TemporalDifferenceQValPolicy
from examples.gridworld.TestRigs.GridFactory import GridFactory
from examples.gridworld.GridWorldState import GridWorldState
from examples.gridworld.GridWorldAgent import GridWorldAgent


class TestTemporalDifferenceQValPolicy(unittest.TestCase):
    __lg = None

    #
    # Load csv (
    #
    # Test level bootstrap of common elements
    #
    def setUp(self):
        self.__lg = logging.getLogger(self.__class__.__name__)
        self.__lg.addHandler(logging.NullHandler)
        self.__tdqvpp = TemporalDifferenceQValPolicyPersistance()
        self.__agent_name = "TestAgent-TemporalDifferenceQValPolicy"
        self.__agent_id = 3142
        self.__exploration_strategy = None  # Not needed for these test cases.

    #
    # Test Greedy Policy
    #
    def test_greedy_policy(self):
        #
        # Load the saved (known) set of Q Vals. See associated XLS file for
        # visualisation of this test set. This is a 20 by 20 GridWorld test
        # case. With one reward (episode exit) and one penalty. The case here
        # is from when the Q Values are very stable and we should see a direct
        # run from the 20,20 start point to the goal in 6,6 going around the
        # penalty at 14,14
        #

        # Set Up
        agent = GridWorldAgent(self.__agent_id, self.__agent_name, self.__exploration_strategy, self.__lg)
        grid = GridFactory.test_grid_four()  # Create grid that matches the 20 by 20 test case.
        state = GridWorldState(grid, agent)

        tdavp = TemporalDifferenceQValPolicy(self.__lg)
        qvals = self.__tdqvpp.load("./greedy_policy_test_1.pb")

        action = tdavp.select_action(agent.name(), state, grid.allowable_actions())
        return


#
# Execute the Test Suite.
#

if __name__ == "__main__":
    tests = TestTemporalDifferenceQValPolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
