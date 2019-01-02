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

random.seed(42)
np.random.seed(42)

itr = 20000
lg = EnvironmentLogging("OneVarParabolicAgent", "OneVarParabolicAgent.log", logging.DEBUG).get_logger()

load = False

pp = GeneralModelParams([[ModelParams.epsilon, float(1)],
                         [ModelParams.epsilon_decay, float(0)],
                         [ModelParams.num_actions, int(2)],
                         [ModelParams.model_file_name, 'OneVarParabolic-ActorCritic'],
                         [ModelParams.verbose, int(0)],
                         [ModelParams.num_states, int(5500)]
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
