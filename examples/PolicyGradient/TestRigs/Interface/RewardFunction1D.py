import abc
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

"""
Interface for all reward functions that exist in 1 dimensional state space with 2 actions. 
"""


class RewardFunction1D(metaclass=abc.ABCMeta):
    """ Interface Definition for reward functions in 1 Dimension
    """

    __fig = None
    __plot_pause = 0.0001

    @classmethod
    def state_as_x(cls,
                   state: float) -> np.array:
        """
        Convert a floating point state into a numpy array in format ready to pass to a NN.
        :param state: The floating point state
        :return numpy array [1, none]:
        """
        xs = np.array([state])
        return xs.reshape([1, xs.shape[0]])

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
        :return: Always 1 as this is for 1D reward functions.
        """
        return 1

    @classmethod
    def num_actions(cls) -> int:
        """
        The number of actions
        :return: Always 2 as this is a 1D state space so only 2 directions of state space traversal.
        """
        return 2

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

        xv = []
        yv = []
        for x in np.arange(self.state_min(), self.state_max(), self.state_step()):
            xv.append(x)
            yv.append(self.reward(x))
        ax = self.__fig.gca()
        ax.set_xlabel('X (State)')
        ax.set_ylabel('Y (Reward)')
        ax.set_title('Reward Function')
        ax.plot(xv, yv)
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
