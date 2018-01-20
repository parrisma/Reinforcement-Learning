import random
import numpy as np
from RandomNonLearningAgent import RandomNonLearningAgent
from TicTacToe import TicTacToe

random.seed(42)
np.random.seed(42)

agent_x = RandomNonLearningAgent(1, "X")
agent_o = RandomNonLearningAgent(-1, "O")

game = TicTacToe(agent_x, agent_o)

game.run(50)
