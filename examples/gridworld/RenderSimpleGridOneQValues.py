import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

from examples.gridworld.SimpleGridOne import SimpleGridOne
from reflrn.Interface.RenderQVals import RenderQVals
from reflrn.Interface.State import State


class RenderSimpleGridOneQValues(RenderQVals):
    __i = 0
    __cmap = 'gist_earth'
    __plot_pause = 0.0001
    __sep_s = ''
    __sep_e = ','
    PLOT_SURFACE = 1
    PLOT_WIREFRAME = 2
    PLOT_GRID = 3

    __ = axes3d  # Dummy row to stop the import showing as unused

    def __init__(self,
                 num_rows: int,
                 num_cols: int,
                 do_scale: bool = False,
                 do_plot: bool = False,
                 plot_style: int = PLOT_SURFACE):
        self.__num_rows = num_rows
        self.__num_cols = num_cols
        self.__do_scale = do_scale
        self.__do_plot = do_plot
        self.__fig = None
        self.__plot_style = plot_style
        self.__plot_funcs = {
            self.PLOT_SURFACE: self.__plot_surface,
            self.PLOT_WIREFRAME: self.__plot_wireframe,
            self.PLOT_GRID: self.__plot_grid,
        }
        self.__view_rot_step = 15
        self.__view_rot = 0

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
                    s += self.__sep_s + fmt.format(qgrid[i][j]) + self.__sep_e
                s += "\n"

            if self.__do_plot:
                if self.__i == 10:
                    self.__i = 0
                    self.__plot(qgrid)
                self.__i += 1

        return s

    #
    # Render the defined type of plot in a on blocking way soc that the
    # plot can be updated in place cycle on cycle.
    #
    def __plot(self, grid: np.ndarray):
        self.__plot_funcs[self.__plot_style](grid)
        return

    #
    # Plot as a 3D surface
    #
    def __plot_surface(self, grid: np.ndarray) -> None:
        self.__plot_3d(grid, wireframe=False)

    #
    # Plot as a 3D wire frame
    #
    def __plot_wireframe(self, grid: np.ndarray) -> None:
        self.__plot_3d(grid, wireframe=True)

    #
    # Plot as a 3D surface
    #
    def __plot_3d(self, grid: np.ndarray, wireframe=True):
        if self.__fig is None:
            self.__fig = plt.figure()
        nr, nc = grid.shape
        x = np.arange(0, nc, 1)
        y = np.arange(0, nr, 1)
        X, Y = np.meshgrid(x, y)
        Z = grid[:]
        np.reshape(Z, X.shape)
        ax = self.__fig.gca(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Q Value')
        ax.set_xticks(np.arange(0, self.__num_cols, int(self.__num_cols / 10)))
        ax.set_yticks(np.arange(0, self.__num_rows, int(self.__num_rows / 10)))
        if self.__view_rot_step > 0:
            self.__view_rot += self.__view_rot_step
            self.__view_rot = self.__view_rot_step % 360
        if wireframe:
            ax.plot_wireframe(X, Y, Z, cmap=self.__cmap, rstride=1, cstride=1)
        else:
            ax.plot_surface(X, Y, Z, cmap=self.__cmap, linewidth=0, antialiased=False)
        plt.pause(self.__plot_pause)
        plt.show(block=False)
        plt.gcf().clear()
        return

    #
    # Plot as a 2D colour grid.
    #
    def __plot_grid(self, grid: np.ndarray):
        if self.__fig is None:
            self.__fig = plt.figure()
        ax = self.__fig.gca()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks(np.arange(0, self.__num_cols, int(self.__num_cols / 10)))
        ax.set_yticks(np.arange(0, self.__num_rows, int(self.__num_rows / 10)))
        ax.imshow(grid, cmap=self.__cmap, interpolation='nearest')
        plt.pause(self.__plot_pause)
        plt.show(block=False)
        plt.gcf().clear()
        return
