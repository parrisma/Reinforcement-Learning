import unittest

from examples.tictactoe.TicTacToe import TicTacToe
from .TestAgent import TestAgent


#
# Unit Test Suite for the TicTacToe concrete implementation of an Environment.
#


class TestTicTacToe(unittest.TestCase):

    def test_1(self):
        print("Test for episode complete")

        test_cases = [("", False, False, False, TicTacToe.no_agent()),
                      ("1:0", False, False, False, 1),
                      ("1:0~1:1~1:2~-1:3~-1:4~-1:8", True, True, False, -1),
                      ("1:0~-1:1~1:2~1:3~-1:4~-1:5~-1:6~1:7~-1:8", True, False, True, -1),
                      ("1:0~-1:1~1:2~-1:3~1:4~-1:5~1:6~-1:7~1:8", True, True, True, 1)]

        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        for test_case, expected1, expected2, expected3, expected4 in test_cases:
            ttt = TicTacToe(agent_x, agent_o, None)
            ttt.import_state(test_case)
            self.assertEqual(ttt.episode_complete(), expected1)
            # verify same but where state is supplied.
            tts = ttt.state()
            self.assertEqual(ttt.episode_complete(tts), expected1)

            episode_summary = ttt.attributes()
            self.assertEqual(episode_summary[TicTacToe.attribute_won[0]], expected2)
            self.assertEqual(episode_summary[TicTacToe.attribute_draw[0]], expected3)
            if episode_summary[TicTacToe.attribute_agent[0]] is not None:
                self.assertEqual(episode_summary[TicTacToe.attribute_agent[0]].id(), expected4)
            else:
                self.assertEqual(episode_summary[TicTacToe.attribute_agent[0]], expected4)

        return

    def test_2(self):
        print("Test for environment import / export")
        test_cases = ("", "1:0", "-1:0", "-1:1",
                      "-1:0~1:2~-1:4~1:6~-1:8",
                      "1:0~-1:1~1:2~-1:3~1:4~-1:5~1:6~-1:7~1:8")
        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        for test_case in test_cases:
            ttt = TicTacToe(agent_x, agent_o, None)
            ttt.import_state(test_case)
            self.assertEqual(ttt.export_state(), test_case)

