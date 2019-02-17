from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


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

    def __init__(self):
        """
        Create a set of 4 sub-plots in a column
            Reward Function
            Q Values
            Probabilities
            Training Loss(es)
        """
        self.fig, self.sub = plt.subplots(5)
        self.fig.suptitle('Actor Critic Telemetry')

        self.reward_func = self.sub[0]
        self.reward_func.set_xlabel('X (State)')
        self.reward_func.set_ylabel('Y (Reward)')
        self.reward_func.set_title('Reward Function')

        self.qvals = self.sub[1]
        self.qvals.set_xlabel('X (State)')
        self.qvals.set_ylabel('Y (Q-Values Predicted)')
        self.qvals.set_title('Action Values')
        self.qvals_x2 = self.qvals.twinx()
        self.qvals_x2.set_ylabel('Y (Q-Values Ref Func)')

        self.probs = self.sub[2]
        self.probs.set_xlabel('X (State)')
        self.probs.set_ylabel('Y (Action Probability)')
        self.probs.set_title('Action Probabilities')

        self.loss = self.sub[3]
        self.loss.set_xlabel('X (Training Episode)')
        self.loss.set_ylabel('Y (Actor Loss)')
        self.loss.set_title('Training Loss')
        self.loss_x2 = self.loss.twinx()
        self.loss_x2.set_ylabel('Y (Critic Loss)')

        self.acc = self.sub[4]
        self.acc.set_xlabel('X (State)')
        self.acc.set_ylabel('Y (Accuracy)')
        self.acc.set_title('Action Probabilities')
        self.acc_x2 = self.acc.twinx()
        self.acc_x2.set_ylabel('Y (Exploration)')
        self.show()

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
        plt.pause(self.plot_pause)
        plt.show(block=False)
        return

    def plot_reward_function(self,
                             func: Callable) -> None:
        """
        Render or Update the Sub Plot for Reward Function
        :param func: The reward function that returns x, y and the rewards z as a grid
        """
        self.__plot_surface(fig=self.reward_func,
                            colour_map=cm.get_cmap('ocean'),
                            func=func)
        return

    def plot_qvals_function(self,
                            func: Callable
                            ) -> None:
        """
        Render or Update the Plot for learned Q-values by state
        :param func: A function that returns x, y state values and corresponding NN predictions for Q Values
        """
        self.__plot_surface(fig=self.qvals,
                            colour_map=cm.get_cmap('RdBu'),
                            func=func)
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
                           func:Callable
                           ) -> None:
        """
        Render or Update the sub plot for action probability Distributions by state
        :param func: A function that returns x, y state values and corresponding NN predictions for action probabilities
        """
        self.__plot_surface(fig=self.probs,
                            colour_map=cm.get_cmap('autumn'),
                            func=func)
        return

    def plot_acc_function(self,
                          actor_acc: list = None,
                          critic_acc: list = None,
                          exploration: list = None
                          ) -> None:
        """
        Render or Update the sub plot for actor / critic accuracy as well as exploration factor
        :param actor_acc: Actor Accuracy Time Series
        :param critic_acc: Critic Accuracy Time Series
        :param exploration: Actor exploration factor
        """
        if actor_acc is not None:
            self.acc.cla()
            self.acc.plot(list(range(0, len(actor_acc))), actor_acc, color=self.plt_color(8))
            self.acc.plot(list(range(0, len(critic_acc))), critic_acc, color=self.plt_color(9))
        if critic_acc is not None:
            self.acc_x2.cla()
            self.acc_x2.plot(list(range(0, len(exploration))), exploration, color=self.plt_color(10))
        self.show()
        return

    def __plot_surface(self,
                       fig: plt.figure,
                       colour_map,
                       func: Callable) -> None:
        """
        Render a surface plot for the given function. Where the function returns x values, y values and the
        corresponding z value for every x,y intersection
        :param fig: The matplotlib figure to plot the surface on
        :param colour_map: The matplotlib colour map (cmap) to use
        :param func: The function that returns x, y and z grid
        """
        x, y, grid = func()
        X, Y = np.meshgrid(x, y)
        Z = grid[:]
        np.reshape(Z, X.shape)
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Q Value')
        ax.set_xticks(np.arange(0, x.size, max(1, int(x.size / 10))))
        ax.set_yticks(np.arange(0, y.size, max(1, int(y.size / 10))))
        self.acc.cla()
        if self.wireframe:
            ax.plot_wireframe(X, Y, Z, colour_map, rstride=1, cstride=1)
        else:
            ax.plot_surface(X, Y, Z, colour_map, linewidth=0, antialiased=False)
        plt.show(block=False)
        return
