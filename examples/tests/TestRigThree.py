import logging

from examples.tictactoe.TicTacToe import TicTacToe
from examples.tictactoe.TicTacToeAgent import TicTacToeAgent
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.TemporalDifferenceQValPolicy import TemporalDifferenceQValPolicy

itr = 5000
lg = EnvironmentLogging("TestRig3", "TestRigThree.log", logging.INFO).get_logger()

agent_x = TicTacToeAgent(1,
                         "X",
                         TemporalDifferenceQValPolicy(lg=lg, filename="./qvn_dump.pb", fixed_games=None,
                                                      load_qval_file=True, manage_qval_file=True),
                         epsilon_greedy=1,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

agent_o = TicTacToeAgent(-1,
                         "O",
                         TemporalDifferenceQValPolicy(lg=lg, filename="./qvn_dump.pb", fixed_games=None),
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

game = TicTacToe(agent_x, agent_o, lg)
game.run(itr)
