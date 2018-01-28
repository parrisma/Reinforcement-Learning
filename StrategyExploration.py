import unittest
from ExplorationPlay import ExplorationPlay
from Agent import Agent
from random import randint


#
# Code in basic TicTacToe strategy.
#

class StrategyExploration(ExplorationPlay):

    #
    # This is a pure random play, just pick any of the possible actions.
    # We cannot see the board, we can only see what actions remain.
    #
    def select_action(self, possible_actions: [int]) -> int:
        # If there is only 1 possible, we have no choice
        if len(possible_actions) == 1:
            return possible_actions[0]

        # Actions to bit mask.
        bm = ~sum(map(lambda n: pow(2, n), possible_actions))

        for a in possible_actions:

            ab = pow(2, a)
            # Complete a row if possible. Either we win or we block
            if bm & 0b000000111 | ab == 0b000000111: return a
            if bm & 0b000111000 | ab == 0b000111000: return a
            if bm & 0b111000000 | ab == 0b111000000: return a

            # Complete a column if possible. Either we win or we block
            if bm & 0b001001001 | ab == 0b001001001: return a
            if bm & 0b010010010 | ab == 0b010010010: return a
            if bm & 0b100100100 | ab == 0b100100100: return a

            # Complete a diagonal if possible. Either we win or we block
            if bm & 0b100010001 | ab == 0b100010001: return a
            if bm & 0b001010100 | ab == 0b001010100: return a

            # Take a corner if < 3 taken. The L strategy
            if bm & 0b100000101 & ab != 0: return a
            if bm & 0b101000100 & ab != 0: return a
            if bm & 0b001000101 & ab != 0: return a
            if bm & 0b101000100 & ab != 0: return a

            # Take the middle.
            if bm & 0b000010000 & ab != 0: return a

        # Take a random
        return possible_actions[randint(0, len(possible_actions) - 1)]

# ********************
# *** UNIT TESTING ***
# ********************


class TestStrategyExploration(unittest.TestCase):

        def test_episode_complete(self):

            test_cases = [([2, 3, 4, 5, 6, 7, 8], 2),
                          ([0, 1, 2, 5, 6, 7, 8], 5),
                          ([0, 1, 2, 3, 4, 5, 8], 8),
                          ([0, 1, 2, 4, 5, 7, 8], 0),
                          ([0, 2, 3, 5, 6, 7, 8], 7),
                          ([0, 1, 2, 3, 5, 6, 7], 0),
                          ([0, 1, 3, 5, 6, 7, 8], 6),
                          ([1, 2, 3, 5, 6, 7, 8], 8),
                          ([0, 1, 2, 3, 5, 7, 8], 2)]

            for possible_actions, expected_action in test_cases:
                se = StrategyExploration()
                self.assertEqual(se.select_action(possible_actions), expected_action)

#
# Execute the tests.
#


if __name__ == "__main__":
    tests = TestStrategyExploration()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
