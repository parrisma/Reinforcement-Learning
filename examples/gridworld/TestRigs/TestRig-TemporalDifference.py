import logging
import random

import numpy as np

from examples.gridworld.GridWorld import GridWorld
from examples.gridworld.GridWorldAgent import GridWorldAgent
from examples.gridworld.SimpleGridOne import SimpleGridOne
from examples.gridworld.SimpleGridOneRenderQValsAsStr import SimpleGridOneRenderQValuesAsStr
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.TemporalDifferencePolicy import TemporalDifferencePolicy

random.seed(42)
np.random.seed(42)

#
# Set Manually and re-run
#
epgrdy = 0.5
itr = 2000
lg = EnvironmentLogging("TestRig-TemporalDifference", "TestRig-TemporalDifference.log", logging.DEBUG).get_logger()

agent_x = GridWorldAgent(1,
                         "G",
                         TemporalDifferencePolicy(lg=lg,
                                                  filename="./gridworld-tmpr-diff.pb",
                                                  fixed_games=None,
                                                  q_val_render=SimpleGridOneRenderQValuesAsStr(4, 5)
                                                  ),
                         epsilon_greedy=epgrdy,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

step = SimpleGridOne.STEP
fire = SimpleGridOne.FIRE
blck = SimpleGridOne.BLCK
goal = SimpleGridOne.GOAL

grid = [
    [step, fire, goal, step, step],
    [step, blck, blck, fire, step],
    [step, blck, blck, blck, step],
    [step, step, step, step, step]
]
sg1 = SimpleGridOne(3,
                    grid,
                    [3, 0],
                    [0, 2])

game = GridWorld(agent_x, sg1, lg)
game.run(itr)
