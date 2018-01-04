import numpy as np
import random
from PlayTicTacToe import PlayTicTacToe

random.seed(42)
np.random.seed(42)
play = PlayTicTacToe()
play.forget_learning()
print(play.Q_Vals())

# APE = PlayTicTacToe.all_possible_endings('-1:8~1:1~-1:6~1:3~-1:7~1:2~',False)
APE = dict()
APE['-1:8~1:1~-1:6~1:3~-1:7~1:2~']=0
QV = play.train_Q_values(len(APE),PlayTicTacToe.moves_to_dict(APE))
QV = play.train_Q_values(len(APE),PlayTicTacToe.moves_to_dict(APE))
QV = play.train_Q_values(len(APE),PlayTicTacToe.moves_to_dict(APE))
QV = play.train_Q_values(len(APE),PlayTicTacToe.moves_to_dict(APE))
QV = play.train_Q_values(len(APE),PlayTicTacToe.moves_to_dict(APE))
QV = play.train_Q_values(len(APE),PlayTicTacToe.moves_to_dict(APE))
print(play.Q_Vals())

print("end")
