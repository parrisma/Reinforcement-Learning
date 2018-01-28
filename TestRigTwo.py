import random
import numpy as np
from TicTacToeAgent import TicTacToeAgent
from HumanPolicy import HumanPolicy
from TemporalDifferencePolicy import TemporalDifferencePolicy
from TicTacToe import TicTacToe

random.seed(42)
np.random.seed(42)

agent_x = TicTacToeAgent(1, "X", TemporalDifferencePolicy(filename="./qvn_dump.pb", load_file=True))
# agent_o = TicTacToeAgent(-1, "O", TemporalDifferencePolicy(filename="./qvn_dump.pb"), 0.5)
agent_o = TicTacToeAgent(-1, "O", HumanPolicy("O"))

game = TicTacToe(agent_x, agent_o)

game.run(50000)
