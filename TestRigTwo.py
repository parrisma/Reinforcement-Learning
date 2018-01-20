import random
import numpy as np
from TicTacToeAgent import TicTacToeAgent
from SimpleRandomPolicy import SimpleRandomPolicy
from TicTacToe import TicTacToe

random.seed(42)
np.random.seed(42)

actions = TicTacToe.actions()
agent_x = TicTacToeAgent(1, "X", SimpleRandomPolicy(actions))
agent_o = TicTacToeAgent(-1, "O", SimpleRandomPolicy(actions))

game = TicTacToe(agent_x, agent_o)

game.run(50)
