import logging
import random

import numpy as np

from examples.tictactoe.TicTacToe import TicTacToe
from examples.tictactoe.TicTacToeAgent import TicTacToeAgent
from reflrn.ActorCriticPolicy import ActorCriticPolicy
from reflrn.DequeReplayMemory import DequeReplayMemory
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.GeneralModelParams import GeneralModelParams
from reflrn.Interface.ModelParams import ModelParams
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.SimpleRandomPolicyWithReplayMemory import SimpleRandomPolicyWithReplayMemory
from reflrn.HumanPolicy import HumanPolicy

random.seed(42)
np.random.seed(42)

itr = 300000
lg = EnvironmentLogging("ActorCriticTicTacToe", "ActorCriticTicTacToe.log", logging.INFO).get_logger()

pp = GeneralModelParams([[ModelParams.epsilon, float(1)],
                         [ModelParams.epsilon_decay, float(0)],
                         [ModelParams.num_actions, int(9)],
                         [ModelParams.model_file_name, 'TicTacToe-ActorCritic']
                         ])

acp = ActorCriticPolicy(policy_params=pp,
                        lg=lg)

agent_x = TicTacToeAgent(1,
                         "X",
                         acp,
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

srp = SimpleRandomPolicyWithReplayMemory(lg, DequeReplayMemory(lg, 500))
agent_o = TicTacToeAgent(-1,
                         "O",
                         srp,
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

game = TicTacToe(agent_x, agent_o, lg)
acp.link_to_env(game)
srp.link_to_env(game)
game.run(itr)

lg.level = logging.DEBUG
itr = 10
hum = HumanPolicy("o", lg)
agent_h = TicTacToeAgent(-1,
                         "O",
                         hum,
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

game2 = TicTacToe(agent_x, agent_h, lg)
hum.link_to_env(game2)
game2.run(itr)
