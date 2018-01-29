import random
import numpy as np
import logging
from TicTacToeAgent import TicTacToeAgent
from HumanPolicy import HumanPolicy
from TemporalDifferencePolicy import TemporalDifferencePolicy
from PureRandomExploration import PureRandomExploration
from StrategyExploration import StrategyExploration
from TicTacToe import TicTacToe
from EnvironmentLogging import EnvironmentLogging

#random.seed(42)
#np.random.seed(42)

#
# Set Manually and re-run
#
learn_mode = False
if not learn_mode:
    epgrdy = 0
    itr = 10
    lg = EnvironmentLogging("TestRig2", "TestRigTwo.log", logging.DEBUG).get_logger()
else:
    epgrdy = 0.5
    itr = 50000
    lg = EnvironmentLogging("TestRig2", "TestRigTwo.log", logging.INFO).get_logger()


agent_x = TicTacToeAgent(1,
                         "X",
                         TemporalDifferencePolicy(lg=lg, filename="./qvn_dump.pb", load_file=True),
                         epsilon_greedy=epgrdy,
                         exploration_play=StrategyExploration(),
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
                             TemporalDifferencePolicy(lg=lg, filename="./qvn_dump.pb"),
                             epsilon_greedy=epgrdy,
                             exploration_play=PureRandomExploration(),
                             lg=lg)

game = TicTacToe(agent_x, agent_o, lg)
game.run(itr)
