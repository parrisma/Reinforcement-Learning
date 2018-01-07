import numpy as np
import random
from PlayTicTacToe import PlayTicTacToe
from pathlib import Path

data_file = "c:/temp/qv_dump.pb"
my_file = Path(data_file)

random.seed(42)
np.random.seed(42)
play = PlayTicTacToe()
play.forget_learning()
print(play.Q_Vals())

if my_file.is_file():
    play.forget_learning()
    play.load_q_vals("c:/temp/qv_dump.pb")
else:
    QV = play.train_Q_values_R(50000)
    print(len(play.Q_Vals()))
    play.save_q_vals(data_file)

#for k,v in QV.items():
#    if(str(QV[k])!=str(QV2[k])):
#        print("Bad Load")
#        break

print(play.Q_Vals_for_state("-1000000000"))
print(play.Q_Vals_for_state("1000000000"))

for i in range(1, 10):
    play.interactive_game()
    play.save_q_vals(data_file)

print("end")