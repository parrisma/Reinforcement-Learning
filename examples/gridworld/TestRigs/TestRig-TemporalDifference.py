import logging
import random

import numpy as np

from examples.gridworld.GridWorld import GridWorld
from examples.gridworld.GridWorldAgent import GridWorldAgent
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.TemporalDifferencePolicy import TemporalDifferencePolicy

random.seed(42)
np.random.seed(42)

#
# Set Manually and re-run
#
learn_mode = True
if not learn_mode:
    epgrdy = 0
    itr = 100
    lg = EnvironmentLogging("TestRig-TemporalDifference", "TestRig-TemporalDifference.log", logging.DEBUG).get_logger()
else:
    epgrdy = 1.0
    itr = 500
    lg = EnvironmentLogging("TestRig-TemporalDifference", "TestRig-TemporalDifference.log", logging.INFO).get_logger()

agent_x = GridWorldAgent(1,
                         "G",
                         TemporalDifferencePolicy(lg=lg,
                                                  filename="./gridworld-tmpr-diff.pb",
                                                  fixed_games=None,
                                                  ),
                         epsilon_greedy=epgrdy,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

game = GridWorld(agent_x, lg)
game.run(itr)
