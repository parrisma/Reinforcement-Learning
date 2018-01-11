import numpy as np
import random
from PlayTicTacToe import PlayTicTacToe
from Persistance import Persistance
from pathlib import Path

data_file = "./qv_dump.pb"
my_file = Path(data_file)

random.seed(42)
np.random.seed(42)

p = Persistance()
x, y = p.load_as_X_Y("./qv_dump.pb")


play = PlayTicTacToe(Persistance())
play.forget_learning()
print(play.q_vals())

if my_file.is_file():
    play.forget_learning()
    play.load_q_vals("./qv_dump.pb")
else:
    QV = play.train_q_values_r(50000)
    print(len(play.q_vals()))
    play.save_q_vals(data_file)

play.play_many(1000)

#for k,v in QV.items():
#    if(str(QV[k])!=str(QV2[k])):
#        print("Bad Load")
#        break

print(play.q_vals_for_state("-1000000000"))
print(play.q_vals_for_state("1000000000"))

human_first = True
for i in range(1, 10):
    play.interactive_game(human_first)
    play.save_q_vals(data_file)
    human_first = not human_first

print("end")