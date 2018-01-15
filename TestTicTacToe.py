import unittest
from TicTacToe import TicTacToe

#
# Unit Test Suite for the TicTacToe concrete implementation of an Environment.
#


class TestTicTacToe(unittest.TestCase):

    def test_episode_complete(self):
        print("Test for episode complete")
        ttt = TicTacToe()

        self.assertEqual(ttt.episode_complete()[TicTacToe.sumry_won], False)
        self.assertEqual(ttt.episode_complete()[TicTacToe.sumry_draw], False)


#
# Execute the tests.
#
if __name__ == "__main__":
    tests = TestTicTacToe()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
