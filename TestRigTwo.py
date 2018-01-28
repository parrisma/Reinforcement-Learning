import random
import logging
import sys
import numpy as np
from TicTacToeAgent import TicTacToeAgent
from HumanPolicy import HumanPolicy
from TemporalDifferencePolicy import TemporalDifferencePolicy
from TicTacToe import TicTacToe

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./TestRigTwo.log',
                    filemode='w')

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
lg = logging.getLogger('TestRigTwo')
lg.addHandler(console)

random.seed(42)
np.random.seed(42)

agent_x = TicTacToeAgent(1, "X", TemporalDifferencePolicy(lg=lg, filename="./qvn_dump.pb", load_file=True), epsilon_greedy=0.5, lg=lg)
agent_o = TicTacToeAgent(-1, "O", TemporalDifferencePolicy(lg=lg), epsilon_greedy=0.5, lg=lg)
if False:
    agent_o = TicTacToeAgent(-1, "O", HumanPolicy("O"), epsilon_greedy=0, lg=lg)

game = TicTacToe(agent_x, agent_o, lg)

game.run(50000)
