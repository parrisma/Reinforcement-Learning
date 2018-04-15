import random

import numpy as np
from examples.tictactoe.PlayTicTacToe import PlayTicTacToe
from QValModel import QValModel

model_filename = "model.h5"

random.seed(42)
np.random.seed(42)

qvm = QValModel()
qvm.load(model_filename)

play = PlayTicTacToe()

print(play.informed_action("-1000000000", False, qvm))
print(play.informed_action("1000000000", False, qvm))

human_first = True
for i in range(1, 10):
    play.interactive_game(human_first, qvm)
    human_first = not human_first

print("end")
