import abc
from typing import Tuple

import numpy as np

"""
Interface for all reward functions that exist in 1 dimensional state space with 2 actions. 
"""


class RewardFunction1D(metaclass=abc.ABCMeta):
    """ Interface Definition for reward functions in 1 Dimension
    """

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

    def state_min(self):
        """
        What is the minimum value of 1D state space
        :return: Minimum value of 1D state space
        """
        raise NotImplementedError

    def state_max(self):
        """
        What is the maximum value of 1D state space
        :return: Maximum value of 1D state space
        """
        raise NotImplementedError

    def state_step(self):
        """
        What is the discrete step increment used to traverse state space (by actions)
        :return: The discrete step increment used to traverse state space (by actions)
        """
        raise NotImplementedError
