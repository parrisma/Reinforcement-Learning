import abc
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

"""
Interface for all reward functions that exist in 2 (state space) dimensional state space with 4 actions. 
"""


class RewardFunction2D(metaclass=abc.ABCMeta):
    """ Interface Definition for reward functions in 2 (state space) Dimensions
    """

    __fig = None
    __plot_pause = 0.0001
    __wireframe = False

    @classmethod
    def state_as_x(cls,
                   state: Tuple[float, float]) -> np.array:
        """
        Convert a floating point state (List/Tuple) into a numpy array in format ready to pass to a NN.
        :param state: The floating point state as 2 element List or Tuple
        :return numpy array [2, none]:
        """
        return np.array([state[0], state[1]])

    def reset(self) -> np.array:
        """
        Reset after an episode has ended.
        :return: The state after reset was performed.
        """
        raise NotImplementedError

    @classmethod
    def state_space_size(cls) -> int:
        """
        The dimensions of the state space
        :return: Always 2 as this is for 2D reward functions. (2 degrees of state space freedom)
        """
        return 2

    @classmethod
    def num_actions(cls) -> int:
        """
        The number of actions
        :return: Always 4 as this is a 2D state space so only 4 directions of state space traversal.
        """
        return 4

    def reward(self,
               state: float) -> float:
        """
        The reward for transitioning to the given state
        :param state: The state being traversed to
        :return: The reward
        """
        raise NotImplementedError

    def step(self,
             actn: int) -> Tuple[np.array, float, bool]:
        """
        Execute the given action
        :param actn: The action to perform.
        :return: State After Action Taken, Reward and Boolean that is True if episode has ended.
        """
        raise NotImplementedError

    def state_min(self) -> float:
        """
        What is the minimum value of 1D state space
        :return: Minimum value of 1D state space
        """
        raise NotImplementedError

    def state_max(self) -> float:
        """
        What is the maximum value of 1D state space
        :return: Maximum value of 1D state space
        """
        raise NotImplementedError

    def state_step(self) -> float:
        """
        What is the discrete step increment used to traverse state space (by actions)
        :return: The discrete step increment used to traverse state space (by actions)
        """
        raise NotImplementedError

    def plot(self) -> None:
        """
        Show a plot of the reward function between the max and min states
        :return: None
        """
        if self.__fig is None:
            self.__fig = plt.figure()

        grid = np.array()
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
        ax.set_xticks(np.arange(0, self.__num_cols, max(1, int(self.__num_cols / 10))))
        ax.set_yticks(np.arange(0, self.__num_rows, max(1, int(self.__num_rows / 10))))
        if self.__view_rot_step > 0:
            self.__view_rot += self.__view_rot_step
            self.__view_rot = self.__view_rot_step % 360
        if self.__wireframe:
            ax.plot_wireframe(X, Y, Z, cmap=self.__cmap, rstride=1, cstride=1)
        else:
            ax.plot_surface(X, Y, Z, cmap=self.__cmap, linewidth=0, antialiased=False)
        plt.pause(self.__plot_pause)
        plt.show(block=False)
        plt.gcf().clear()
        plt.pause(self.__plot_pause)
        plt.show(block=False)
        return

    def func(self) -> Tuple[list, list]:
        """
        :return: The X and Y values for the function between the min and max state values at the defined step
        """
        xv = []
        yv = []
        for x in np.arange(self.state_min(), self.state_max(), self.state_step()):
            xv.append(x)
            yv.append(self.reward(x))
        return xv, yv
