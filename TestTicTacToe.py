import unittest
from TicTacToe import TicTacToe

#
# Unit Test Suite for the TicTacToe concrete implementation of an Environment.
#


class TestTicTacToe(unittest.TestCase):

    def test_episode_complete(self):
        print("Test for episode complete")

        test_cases = [("", False),
                      ("1:1", False),
                      ("1:1~1:2~1:3~-1:4~-1:5~-1:9", True),
                      ("1:1~-1:2~1:3~-1:4~1:5~-1:6~1:7~-1:8~1:9", True)]

        for test_case, expected in test_cases:
            ttt = TicTacToe()
            ttt.import_state(test_case)
            self.assertEqual(ttt.episode_complete(), expected)

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
