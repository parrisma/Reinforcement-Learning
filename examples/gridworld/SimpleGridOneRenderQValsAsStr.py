import numpy as np

from examples.gridworld.SimpleGridOne import SimpleGridOne
from reflrn.RenderQValsAsStr import RenderQValsAsStr
from reflrn.State import State


class SimpleGridOneRenderQValuesAsStr(RenderQValsAsStr):

    def __init__(self,
                 num_rows: int,
                 num_cols):
        self.__num_rows = num_rows
        self.__num_cols = num_cols

    def render_as_str(self, curr_state: State, q_vals: dict) -> str:
        qgrid = np.zeros((self.__num_rows, self.__num_cols))
        s = ""
        if q_vals is not None:
            for k in q_vals:
                r, c = [(lambda x: int(x))(x) for x in k.split(',')]
                for kqv in q_vals[k]:
                    x, y = SimpleGridOne.coords_after_action(r, c, kqv)
                    if qgrid[x][y] == np.float(0):
                        qgrid[x][y] = (q_vals[k])[kqv]
                    else:
                        qgrid[x][y] += (q_vals[k])[kqv]
                        qgrid[x][y] /= np.float(2)

            for i in range(0, self.__num_rows):
                for j in range(0, self.__num_cols):
                    s += "[" + '{:+.16f}'.format(qgrid[i][j]) + "] "
                s += "\n"
        return s
