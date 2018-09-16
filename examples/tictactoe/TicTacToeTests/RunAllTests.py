import unittest

from examples.tictactoe.TicTacToeTests.TestTicTacToe import TestTicTacToe

#
# Execute the TicTacToe Unit Test Suite.
#
if __name__ == "__main__":
    tests = TestTicTacToe()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
