import logging
import random

import numpy as np
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.ModelParameters import ModelParameters
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.DequeReplayMemory import DequeReplayMemory
from reflrn.SimpleRandomPolicyWithReplayMemory import SimpleRandomPolicyWithReplayMemory
from reflrn.TemporalDifferenceActorCriticDeepNNPolicy import TemporalDifferenceActorCriticDeepNNPolicy
from reflrn.TicTacToe import TicTacToe
from reflrn.TicTacToeAgent import TicTacToeAgent

random.seed(42)
np.random.seed(42)

itr = 500000
lg = EnvironmentLogging("TestRig5", "TestRigFive.log", logging.INFO).get_logger()

replay_memory = DequeReplayMemory(lg, 1000)

fn = "keras.model.critic"
tdacdnnp = TemporalDifferenceActorCriticDeepNNPolicy(lg=lg,
                                                     replay_memory=replay_memory,
                                                     model_file_name=fn,
                                                     model_parameters=ModelParameters(),
                                                     load_model=True)

agent_x = TicTacToeAgent(1,
                         "X",
                         tdacdnnp,
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

agent_o = TicTacToeAgent(-1,
                         "O",
                         SimpleRandomPolicyWithReplayMemory(lg, replay_memory),
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

game = TicTacToe(agent_x, agent_o, lg)
game.run(itr)
