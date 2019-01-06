import logging
import random

import numpy as np

from examples.PolicyGradient.OneVarParabolicAgent import OneVarParabolicAgent
from examples.PolicyGradient.OneVarParabolicEnv import OneVarParabolicEnv
from examples.PolicyGradient.OneVarParabolicNN import OneVarParabolicNN
from reflrn.ActorCriticPolicyTDQVal import ActorCriticPolicyTDQVal
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.GeneralModelParams import GeneralModelParams
from reflrn.Interface.ModelParams import ModelParams
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.SimpleLearningRate import SimpleLearningRate

random.seed(42)
np.random.seed(42)

itr = 200
lg = EnvironmentLogging("OneVarParabolicAgent", "OneVarParabolicAgent.log", logging.DEBUG).get_logger()

load = False

learning_rate_0 = float(1)

pp = GeneralModelParams([[ModelParams.epsilon, float(1)],
                         [ModelParams.epsilon_decay, float(0)],
                         [ModelParams.num_actions, int(2)],
                         [ModelParams.model_file_name, 'OneVarParabolic-ActorCritic'],
                         [ModelParams.verbose, int(0)],
                         [ModelParams.num_states, int(20)],
                         [ModelParams.learning_rate_0, learning_rate_0],
                         [ModelParams.learning_rate_decay, SimpleLearningRate.lr_decay_target(learning_rate_0,
                                                                                              int(itr / 2),
                                                                                              float(0.1))],
                         [ModelParams.learning_rate_min, float(0)]
                         ])

nn = OneVarParabolicNN(1,  # One variable
                       pp.get_parameter(ModelParams.num_actions))

acp = ActorCriticPolicyTDQVal(policy_params=pp,
                              network=nn,
                              lg=lg)
if load:
    acp.load('OneVarParabolic')

agent_x = OneVarParabolicAgent(1,
                               "X",
                               acp,
                               epsilon_greedy=0,
                               exploration_play=PureRandomExploration(),
                               lg=lg)
agent_x.explain = False

if not load:
    game = OneVarParabolicEnv(agent=agent_x,
                              lg=lg,
                              explain=False)
    acp.link_to_env(game)
    acp.explain = False
    game.run(itr)
