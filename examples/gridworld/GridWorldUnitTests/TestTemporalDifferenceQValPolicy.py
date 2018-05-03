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

        tdavp = TemporalDifferenceQValPolicy(lg=self.__lg,
                                             filename="./greedy_policy_test_1.pb",
                                             load_qval_file=True)

        # ToDo
        # Test not finished, but manual stepping has shown it gets stuck in a tight cell to cell loop
        # if just the greedy policy is followed. Not sure why this is as in learning mode this should
        # reduce the QValue score as the step cost is -1 and it would then select another max QVal from
        # adjacent cell as new optimal ??
        #
        # The issue is that at this advanced stage (learning rate is very small) so updates are order
        # -1e-06, so the impact on the QValue is minimal order < 100K iterations, by which time the
        # random exploration move has happened. This means when we get stuck in loops we tend to have very
        # long episodes while waiting to break out of these tight loops.
        #
        expected_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        i = 0
        while not grid.episode_complete():  # i < len(expected_actions):
            print(state.state_as_string())
            action = tdavp.select_action(agent.name(), state, grid.allowable_actions())
            grid.execute_action(action)
            state = GridWorldState(grid, agent)
            # self.assertEqual(expected_actions[i], action)
            i += 1
        return


#
# Execute the Test Suite.
#

if __name__ == "__main__":
    tests = TestTemporalDifferenceQValPolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
