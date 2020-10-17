import unittest

import numpy as np

from examples.tictactoe.TicTacToe import TicTacToe
from examples.tictactoe.TicTacToeState import TicTacToeState
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
                      ("1:0~-1:1~1:2~-1:3~1:4~-1:5~1:6~-1:7~1:8", True, True, False, 1)]

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

    def test_tic_tac_toe_state(self):
        ao = TestAgent(1, "O")
        ax = TestAgent(-1, "X")
        id_o = ao.id()
        id_x = ax.id()
        test_cases = [(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])),
                      (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                      (np.array([id_o, id_o, id_o, id_o, id_o, id_o, id_o, id_o, id_o])),
                      (np.array([id_x, id_x, id_x, id_x, id_x, id_x, id_x, id_x, id_x])),
                      (np.array([id_x, id_o, id_x, id_o, id_x, id_o, id_x, id_o, id_x])),
                      (np.array([id_o, id_x, id_o, id_x, id_o, id_x, id_o, id_x, id_o])),
                      (np.array([id_x, 0, id_o, id_x, 0, id_x, np.inf, np.nan, id_o])),
                      (np.array([[id_x, 0, id_o], [id_x, 0, id_x], [np.inf, np.nan, id_o]]))
                      ]
        expected_inv = [(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])),
                        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                        (np.array([id_x, id_x, id_x, id_x, id_x, id_x, id_x, id_x, id_x])),
                        (np.array([id_o, id_o, id_o, id_o, id_o, id_o, id_o, id_o, id_o])),
                        (np.array([id_o, id_x, id_o, id_x, id_o, id_x, id_o, id_x, id_o])),
                        (np.array([id_x, id_o, id_x, id_o, id_x, id_o, id_x, id_o, id_x])),
                        (np.array([id_o, 0, id_x, id_o, 0, id_o, np.inf, np.nan, id_x])),
                        (np.array([[id_o, 0, id_x], [id_o, 0, id_o], [np.inf, np.nan, id_x]]))
                        ]
        for case, expected1 in zip(test_cases, expected_inv):
            tts = TicTacToeState(board=case, agent_o=ao, agent_x=ax)
            self.assertTrue(tts.state() is not case)  # we expect State to deep-copy the input
            self.assertTrue(self.__np_eq(tts.state(), case))
            self.assertTrue(self.__np_eq(tts.invert_player_perspective().state(), expected1))
        return

    #
    # Are the given arrays equal shape and element by element content. We allow nan = nan as equal.
    #
    @classmethod
    def __np_eq(cls,
                npa1: np.array,
                npa2: np.array) -> bool:
        if np.shape(npa1) != np.shape(npa2):
            return False
        v1 = np.reshape(npa1, np.size(npa1))
        v2 = np.reshape(npa2, np.size(npa2))

        for vl, vr in np.stack((v1, v2), axis=-1):
            if np.isnan(vl):
                if np.isnan(vl) != np.isnan(vr):
                    return False
            else:
                if vl != vr:
                    return False
        return True
