import logging
from TicTacToeAgent import TicTacToeAgent
from PureRandomExploration import PureRandomExploration
from TicTacToe import TicTacToe
from EnvironmentLogging import EnvironmentLogging
from TemporalDifferenceActorCriticDeepNNPolicy import TemporalDifferenceActorCriticDeepNNPolicy
from SimpleRandomPolicy import SimpleRandomPolicy

itr = 500000
lg = EnvironmentLogging("TestRig5", "TestRigFive.log", logging.INFO).get_logger()


fn = "keras.model.critic"
tdacdnnp = TemporalDifferenceActorCriticDeepNNPolicy(lg=lg, model_file_name=fn, load_model=True)

agent_x = TicTacToeAgent(1,
                         "X",
                         tdacdnnp,
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

agent_o = TicTacToeAgent(-1,
                         "O",
                         SimpleRandomPolicy(),
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

game = TicTacToe(agent_x, agent_o, lg)
game.run(itr)
