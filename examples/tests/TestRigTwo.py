import logging
import random

import numpy as np

from examples.tictactoe.TicTacToe import TicTacToe
from examples.tictactoe.TicTacToeAgent import TicTacToeAgent
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.HumanPolicy import HumanPolicy
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.TemporalDifferenceQValPolicy import TemporalDifferenceQValPolicy

random.seed(42)
np.random.seed(42)

#
# Set Manually and re-run
#
learn_mode = True
if not learn_mode:
    epgrdy = 0
    itr = 100
    lg = EnvironmentLogging("TestRig2", "TestRigTwo.log", logging.DEBUG).get_logger()
else:
    epgrdy = 1.0
    itr = 500000
    lg = EnvironmentLogging("TestRig2", "TestRigTwo.log", logging.INFO).get_logger()

agent_x = TicTacToeAgent(1,
                         "X",
                         TemporalDifferenceQValPolicy(lg=lg, filename="./qvn_dump.pb", fixed_games=None,
                                                      load_qval_file=True, manage_qval_file=True),
                         epsilon_greedy=epgrdy,
                         exploration_play=PureRandomExploration(),
                         lg=lg)
if not learn_mode:
    agent_o = TicTacToeAgent(-1,
                             "O", HumanPolicy("O", lg=lg),
                             epsilon_greedy=0,
                             exploration_play=PureRandomExploration(),
                             lg=lg)
else:
    agent_o = TicTacToeAgent(-1,
                             "O",
                             TemporalDifferenceQValPolicy(lg=lg, filename="./qvn_dump.pb", fixed_games=None, ),
                             epsilon_greedy=epgrdy,
                             exploration_play=PureRandomExploration(),
                             lg=lg)

game = TicTacToe(agent_x, agent_o, lg)
game.run(itr)
