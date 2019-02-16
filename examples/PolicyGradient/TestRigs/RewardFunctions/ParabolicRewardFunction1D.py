from typing import Tuple

import numpy as np

from examples.PolicyGradient.TestRigs.Interface.RewardFunction1D import RewardFunction1D

"""
 This is the environment that returns rewards. The reward profile is -x^2 where the state is
 x between -1 and 1 on 0.05 step intervals. This means the optimal action to maximise reward
 is to always move such that x approaches 0. The actions are 0 = move right on x axis and 1
 move left on x axis. -0.05, 0.00 & 0.05 all return zero reward such that the agent should tend
 to move such that it stays in this region.

 The environment is episodic such that if actor moves below -1 or above 1 the episode restarts.

"""


class ParabolicRewardFunction1D(RewardFunction1D):
    __state_min = float(-1)
    __state_max = float(1)
    __state_step = 0.05

    def __init__(self):
        """
        Start in a default reset state.
        """
        self.state = None
        self.reset()
        return

    def reset(self) -> np.array:
        """
        Reset state to either extreme of state space.
        :return: The state after reset was performed.
        """
        if np.random.rand() >= 0.5:
            self.state = self.__state_max
        else:
            self.state = self.__state_min
        return np.array([self.state])

    def reward(self,
               state: float) -> float:
        """
        Compute the reward for the given state; which for this reward function is
        simply -(x^2). Except in the range +/- one step around the turning point
        when the reward is hard wired to zero.
        :param state:
        :return: Reward for given state
        """
        if -self.__state_step <= state <= self.__state_step:
            return float(0)
        return -(state * state)

    def step(self,
             actn: int) -> Tuple[np.array, float, bool]:
        """
        Take the specified action
        :param actn: the action to take
        :return: The new state, the reward for the state transition and bool, which is true if episode ended
        """
        if actn == 0:
            self.state += self.__state_step
        elif actn == 1:
            self.state -= self.__state_step
        else:
            raise ValueError("Action can only be value 0 or 1 so [" + str(actn) + "] is illegal")

        self.state = np.round(self.state, 3)
        dn = (self.state < self.__state_min or self.state > self.__state_max)

        return np.array([self.state]), self.reward(self.state), dn

    @classmethod
    def state_space_dimension(cls) -> int:
        """
        The dimensions of the state space
        :return: Always 1 as this is for 1D reward functions.
        """
        return super(ParabolicRewardFunction1D, cls).state_space_dimension()

    @classmethod
    def num_actions(cls) -> int:
        """
        The number of actions. Always 2 as this is a 1D state space so only 2 directions of state space traversal.
        East and West
        :return: Number of actions as integer.
        """
        return super(ParabolicRewardFunction1D, cls).num_actions()

    def state_min(self) -> float:
        """
        What is the minimum value of 1D state space
        :return: Minimum value of 1D state space
        """
        return self.__state_min

    def state_max(self) -> float:
        """
        What is the maximum value of 1D state space
        :return: Maximum value of 1D state space
        """
        return self.__state_max

    def state_step(self) -> float:
        """
        What is the discrete step increment used to traverse state space (by actions)
        :return: The discrete step increment used to traverse state space (by actions)
        """
        return self.__state_step

    def state_shape(self) -> Tuple[int, int]:
        """
        What are the dimensions (Shape) of the state space
        :return: Tuple describing the shape
        """
        return super(ParabolicRewardFunction1D, self).state_shape()
