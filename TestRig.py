import numpy as np
import random
from PlayTicTacToe import PlayTicTacToe

random.seed(42)
np.random.seed(42)
play = PlayTicTacToe()
play.forget_learning()
print(play.Q_Vals())

QV = play.train_Q_values_R(5)
print(play.Q_Vals())
play.save_q_vals("./dumps/qv_dump.pb")
play.forget_learning()
print(play.Q_Vals())
play.load_q_vals("./dumps/qv_dump.pb")
print(play.Q_Vals())

print(play.Q_Vals_for_state("-1000000000"))
print(play.Q_Vals_for_state("1000000000"))
GI = {}
GR = {}
GD = {}
GI,GR,GD = play.play_many(1000)
play.game().reset()
GI = {}
GR = {}
GD = {}
for i in range(1,10):
    play.interactive_game()

print("end")