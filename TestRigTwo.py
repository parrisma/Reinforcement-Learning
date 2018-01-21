import random
import numpy as np
from TicTacToeAgent import TicTacToeAgent
#  from SimpleRandomPolicy import SimpleRandomPolicy
from TemporalDifferencePolicy import TemporalDifferencePolicy
from TicTacToe import TicTacToe

random.seed(42)
np.random.seed(42)

#  agent_x = TicTacToeAgent(1, "X", SimpleRandomPolicy())
#  agent_o = TicTacToeAgent(-1, "O", SimpleRandomPolicy())

agent_x = TicTacToeAgent(1, "X", TemporalDifferencePolicy("./qvn_dump.pb"))
agent_o = TicTacToeAgent(-1, "O", TemporalDifferencePolicy("./qvn_dump.pb"))

game = TicTacToe(agent_x, agent_o)

game.run(5000)
