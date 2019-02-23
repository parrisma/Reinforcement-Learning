import math
from typing import Callable, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D - this is needed but shows as unused - registers 3d projection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Visualise2D:
    fig = None
    sub = None
    reward_func = None
    qvals = None
    qvals_x2 = None
    probs = None
    loss = None
    loss_x2 = None
    plot_pause = 0.0001
    nxt_col = 0
    plt_col = dict()
    wireframe = False
    plot_canvas_x_inches = 15
    plot_canvas_y_inches = 10

    def __init__(self):
        """
        Create a set of 4 sub-plots in a column
            Reward Function
            Q Values
            Probabilities
            Training Loss(es)
        """
        plt.rcParams["figure.figsize"] = (Visualise2D.plot_canvas_x_inches, Visualise2D.plot_canvas_y_inches)
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Telemetry')

        self.reward_func = self.fig.add_subplot(2, 2, 1, projection='3d')
        self.reward_func.set_xlabel('State X')
        self.reward_func.set_ylabel('State Y')
        self.reward_func.set_title('Reward Function')

        self.qvals = self.fig.add_subplot(2, 2, 3)
        self.qvals.set_xlabel('State X')
        self.qvals.set_ylabel('State Y')
        self.qvals.set_title('Predicted Q-Vals')
        self.qvals_cbar_added = False

        self.probs = self.fig.add_subplot(2, 2, 4)
        self.probs.set_xlabel('State X')
        self.probs.set_ylabel('State Y')
        self.probs.set_title('Predicted Actn Probs')
        self.prob_cbar_added = False

        self.loss = self.fig.add_subplot(2, 2, 2)
        self.loss.set_xlabel('Training Episode')
        self.loss.set_ylabel('Actor Loss')
        self.loss.set_title('Training Loss')
        self.loss_x2 = self.loss.twinx()
        self.loss_x2.set_ylabel('Critic Loss')

        return

    def plt_color(self,
                  plt_id: int) -> str:
        """
        :param: pt_id a unique numerical id for the plot with which to associate a color
        A random color code to use for setting line colours for the given plot id
        :return: color in form 'C<n>' where n is a integer that increased by 1 for every call.
        """
        if plt_id not in self.plt_col:
            self.plt_col[plt_id] = "C{:d}".format(self.nxt_col)
            self.nxt_col += 1
        return self.plt_col[plt_id]

    def show(self) -> None:
        """
        Render the Figure + Sub Plots
        """
        self.fig.tight_layout()
        plt.pause(self.plot_pause)
        plt.show(block=False)
        return

    def plot_reward_function(self,
                             func: Callable[[None], Tuple[np.ndarray, np.ndarray, np.ndarray]]
                             ) -> None:
        """
        Render or Update the Sub Plot for Reward Function
        :param func: The reward function that returns x, y and the rewards z as a grid
        """
        self.__plot_surface(ax_in=self.reward_func,
                            colour_map=cm.get_cmap('ocean'),
                            func=func)
        return

    def plot_qvals_function(self,
                            func: Callable[[None], Tuple[np.ndarray, np.ndarray, np.ndarray]]
                            ) -> None:
        """
        Render or Update the Plot for learned Q-values by state
        :param func: A function that returns x, y state values and corresponding NN predictions for Q Values
        """
        self.__plot_prob_surface(ax=self.qvals,
                                 colour_map=cm.get_cmap('Greens'),
                                 func=func,
                                 color_bar_added=self.qvals_cbar_added)
        self.qvals_cbar_added = True
        return

    def plot_loss_function(self,
                           actor_loss: list = None,
                           critic_loss: list = None
                           ) -> None:
        """
        Render or Update the sub plot for actor / critic loss
        :param actor_loss: Actor Loss Time Series
        :param critic_loss: Critic Loss Time Series
        """
        if actor_loss is not None:
            self.loss.cla()
            self.loss.plot(list(range(0, len(actor_loss))), actor_loss, color=self.plt_color(5))
        if critic_loss is not None:
            self.loss_x2.cla()
            self.loss_x2.plot(list(range(0, len(critic_loss))), critic_loss, color=self.plt_color(6))
        self.show()
        return

    def plot_prob_function(self,
                           func: Callable[[None], type([np.ndarray, np.ndarray, np.ndarray])]
                           ) -> None:
        """
        Render or Update the sub plot for action probability by state
        :param func: A function that returns x, y state values and corresponding NN predictions for action probabilities
        """
        self.__plot_prob_surface(ax=self.probs,
                                 colour_map=cm.get_cmap('Reds'),
                                 func=func,
                                 color_bar_added=self.prob_cbar_added)
        self.prob_cbar_added = True
        return

    # def plot_acc_function(self,
    #                      actor_acc: list = None,
    #                      critic_acc: list = None,
    #                      exploration: list = None
    #                      ) -> None:
    #    """
    #    Render or Update the sub plot for actor / critic accuracy as well as exploration factor
    #    :param actor_acc: Actor Accuracy Time Series
    #    :param critic_acc: Critic Accuracy Time Series
    #    :param exploration: Actor exploration factor
    #    """
    #    if actor_acc is not None:
    #        self.acc.cla()
    #        self.acc.plot(list(range(0, len(actor_acc))), actor_acc, color=self.plt_color(8))
    #        self.acc.plot(list(range(0, len(critic_acc))), critic_acc, color=self.plt_color(9))
    #    if critic_acc is not None:
    #        self.acc_x2.cla()
    #        self.acc_x2.plot(list(range(0, len(exploration))), exploration, color=self.plt_color(10))
    #    self.show()
    #    return

    def __plot_surface(self,
                       ax_in: plt.figure,
                       colour_map,
                       func: Callable[[None], Tuple[np.ndarray, np.ndarray, np.ndarray]]
                       ) -> None:
        """
        Render a surface plot for the given function. Where the function returns x values, y values and the
        corresponding z value for every x,y intersection
        :param ax: The matplotlib figure to plot the surface on
        :param colour_map: The matplotlib colour map (cmap) to use
        :param func: The function that returns x, y and z grid
        """
        ax = ax_in

        # Grab the reward function data set
        x, y, grid = func(None)
        _x, _y = np.meshgrid(x, y)
        _z = grid[:]
        _z = np.reshape(_z, _x.shape)

        # Plot the surface.
        surf = ax.plot_surface(_x, _y, _z, cmap=colour_map,
                               linewidth=0, antialiased=False)
        self.fig.colorbar(mappable=surf, ax=ax, shrink=0.75, aspect=20)

        # Customize the z axis.
        ax.set_zlim(math.ceil(np.min(_z) * 1.1), math.ceil(np.max(_z) * 1.1))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_xticks(np.arange(0, x.size, max(1, int(x.size / 10))))
        ax.set_yticks(np.arange(0, y.size, max(1, int(y.size / 10))))

        self.show()
        return

    def __plot_prob_surface(self,
                            ax: plt.figure,
                            colour_map: cm,
                            func: Callable[[None], Tuple[np.ndarray, np.ndarray, np.ndarray]],
                            color_bar_added: bool
                            ) -> None:
        """
        Render a probability map as a (non blocking) 2D plot. The data grid should be the action probabilities for
        traversing the reward surface N,S,E,W
        :param ax: The matplotlib figure to plot the surface on
        :param colour_map: The matplotlib colour map (cmap) to use
        :param func: The function that returns x, y and z grid for the entire state space of the current env.
        :param color_bar_added: Has the color bar already been added to this plot (avoid multi c-bars)
        """
        x, y, grid = func()
        nx = x.shape[0] * 3
        ny = y.shape[0] * 3
        z = np.zeros((nx, ny))

        gx = 0
        gy = 0
        for a in range(1, nx, 3):
            for b in range(1, ny, 3):
                i = a - 1
                j = b - 1
                z[i + 0, j + 1] = grid[gx, gy][0]  # N
                z[i + 1, j + 2] = grid[gx, gy][1]  # E
                z[i + 2, j + 1] = grid[gx, gy][2]  # S
                z[i + 1, j + 0] = grid[gx, gy][3]  # W
                z[i + 1, j + 1] = z[i:i + 3, j:j + 3].sum() / 4.0
                z[i + 0, j + 0] = (z[i + 0, j + 1] + z[i + 1, j + 0] + z[i + 1, j + 1]) / 3.0
                z[i + 2, j + 0] = (z[i + 1, j + 0] + z[i + 2, j + 1] + z[i + 1, j + 1]) / 3.0
                z[i + 2, j + 2] = (z[i + 2, j + 1] + z[i + 1, j + 2] + z[i + 1, j + 1]) / 3.0
                z[i + 0, j + 2] = (z[i + 0, j + 1] + z[i + 1, j + 2] + z[i + 1, j + 1]) / 3.0
                z[i:i + 3, j:j + 3] -= np.max(z[i:i + 3, j:j + 3])
                z[i:i + 3, j:j + 3] = np.fabs(z[i:i + 3, j:j + 3])
                slz = z[i:i + 3, j:j + 3]
                slz /= slz.sum()
                gx += 1
            gy += 1
            gx = 0

        cf = ax.imshow(z, interpolation='nearest', cmap=colour_map)
        if not color_bar_added:
            self.fig.colorbar(cf, ax=ax, shrink=0.75, aspect=20)
        self.show()
        return
