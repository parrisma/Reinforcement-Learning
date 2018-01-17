import unittest
import numpy as np
from TicTacToe import TicTacToe

#
# Unit Test Suite for the TicTacToe concrete implementation of an Environment.
#


class TestTicTacToe(unittest.TestCase):

    def test_episode_complete(self):
        print("Test for episode complete")

        test_cases = [("", False, False, False, TicTacToe.no_actor()),
                      ("1:1", False, False, False, 1),
                      ("1:1~1:2~1:3~-1:4~-1:5~-1:9", True, True, False, -1),
                      ("1:1~-1:2~1:3~1:4~-1:5~-1:6~-1:7~1:8~-1:9", True, False, True, -1),
                      ("1:1~-1:2~1:3~-1:4~1:5~-1:6~1:7~-1:8~1:9", True, True, True, 1)]

        for test_case, expected1, expected2, expected3, expected4 in test_cases:
            ttt = TicTacToe()
            ttt.import_state(test_case)
            self.assertEqual(ttt.episode_complete(), expected1)

            episode_summary = ttt.episode_summary()
            self.assertEqual(episode_summary[TicTacToe.sumry_won], expected2)
            self.assertEqual(episode_summary[TicTacToe.sumry_draw], expected3)
            if np.isnan(expected4):
                self.assertEqual(np.isnan(episode_summary[TicTacToe.sumry_actor]), True)
            else:
                self.assertEqual(episode_summary[TicTacToe.sumry_actor], expected4)

    def test_state_import_export(self):
        print("Test for environment import / export")

        test_cases = ("", "1:1", "-1:1", "-1:2",
                      "-1:1~1:3~-1:5~1:7~-1:9",
                      "1:1~-1:2~1:3~-1:4~1:5~-1:6~1:7~-1:8~1:9")
        for test_case in test_cases:
            ttt = TicTacToe()
            ttt.import_state(test_case)
            self.assertEqual(ttt.export_state(), test_case)


#
# Execute the tests.
#
if __name__ == "__main__":
    tests = TestTicTacToe()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
