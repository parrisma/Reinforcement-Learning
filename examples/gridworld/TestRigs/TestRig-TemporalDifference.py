import logging
import random

import numpy as np

from examples.gridworld.GridWorld import GridWorld
from examples.gridworld.GridWorldAgent import GridWorldAgent
from examples.gridworld.SimpleGridOneRenderQValsAsStr import SimpleGridOneRenderQValues
from examples.gridworld.TestRigs.GridFactory import GridFactory
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.TemporalDifferencePolicy import TemporalDifferencePolicy

random.seed(42)
np.random.seed(42)

#
# Set Manually and re-run
#
epgrdy = 0.5
itr = 3000
lg = EnvironmentLogging("TestRig-TemporalDifference", "TestRig-TemporalDifference.log", logging.DEBUG).get_logger()

test_grid = GridFactory.test_grid_three()
sh = test_grid.shape()

agent_x = GridWorldAgent(1,
                         "G",
                         TemporalDifferencePolicy(lg=lg,
                                                  filename="./gridworld-tmpr-diff.pb",
                                                  fixed_games=None,
                                                  q_val_render=SimpleGridOneRenderQValues(sh[0],
                                                                                          sh[1],
                                                                                          do_scale=True,
                                                                                          do_plot=True)
                                                  ),
                         epsilon_greedy=epgrdy,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

game = GridWorld(agent_x, test_grid, lg)
game.run(itr)
