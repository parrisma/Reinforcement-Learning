import logging

from examples.tictactoe.TicTacToe import TicTacToe
from examples.tictactoe.TicTacToeAgent import TicTacToeAgent
from reflrn.ActorCriticPolicy import ActorCriticPolicy
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.PureRandomExploration import PureRandomExploration
from reflrn.RandomPolicy import RandomPolicy

itr = 5000
lg = EnvironmentLogging("TestRig4", "TestRigFour.log", logging.INFO).get_logger()

acp = ActorCriticPolicy(lg)
agent_x = TicTacToeAgent(1,
                         "X",
                         acp,
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

agent_o = TicTacToeAgent(-1,
                         "O",
                         RandomPolicy(),
                         epsilon_greedy=0,
                         exploration_play=PureRandomExploration(),
                         lg=lg)

game = TicTacToe(agent_x, agent_o, lg)
acp.link_to_env(game)
game.run(itr)
