import matplotlib.pyplot as plt
import numpy as np

from examples.gridworld.SimpleGridOne import SimpleGridOne
from reflrn.RenderQVals import RenderQVals
from reflrn.State import State


class SimpleGridOneRenderQValues(RenderQVals):

    __i = 0

    def __init__(self,
                 num_rows: int,
                 num_cols: int,
                 do_scale: bool = False,
                 do_plot: bool = False):
        self.__num_rows = num_rows
        self.__num_cols = num_cols
        self.__do_scale = do_scale
        self.__do_plot = do_plot

    def render(self, curr_state: State, q_vals: dict) -> str:
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

            if self.__do_scale:
                mn = np.min(qgrid)
                mx = np.max(qgrid)
                for i in range(0, self.__num_rows):
                    for j in range(0, self.__num_cols):
                        qgrid[i][j] = (((qgrid[i][j] - mn) / (mx - mn)) - 0) * 100
                fmt = '{0:.2f}%'
            else:
                fmt = '{:+.16f}'

            for i in range(0, self.__num_rows):
                for j in range(0, self.__num_cols):
                    s += "[" + fmt.format(qgrid[i][j]) + "] "
                s += "\n"

            if self.__do_plot:
                if self.__i == 0:
                    plt.imshow(qgrid, cmap='hot', interpolation='nearest')
                    plt.draw()
                    plt.pause(0.0001)
                    plt.show(block=False)
                self.__i += 1
                if self.__i == 10:
                    self.__i = 0

        return s
