import math
from typing import Tuple

import numpy as np

from examples.PolicyGradient.TestRigs.Interface.RewardFunction1D import RewardFunction1D

"""
This Reward Function has two local maxima and one global maxima. This is modelled as 2.5 cycles
of a sinusoidal curve with the 2nd (central) peek weighed as give it a larger magnitude then the
peeks either side. The function is symmetrical around the turning point of the central peek.

So the RL Agent should be able to find and maximise to move the agent to the central peek even
when starting at the state space extremity and having to pass through the local maxima. 

If x is a value in radians and state space is in range (0 to m)then the reward function is:
   sin(x)* EXP(1-ABS((x-(m/2))/(m/2)))

"""


class LocalMaximaRewardFunction1D(RewardFunction1D):
    __state_min = int(0)
    __state_max = int(60)
    __state_step = int(1)
    __center_state = int(__state_max / 2.0)
    __x_min = float(0)
    __x_max = float(2.5 * (2 * math.pi))
    __x_step = float(15.0 * (math.pi / 180.0))  # 15 degree steps as radians

    def __init__(self):
        """
        Start in a default reset state.
        """
        self.state = None
        self.num_steps = int((self.__state_max - self.__state_min) / self.__state_step)
        self.reset()
        return

    def reset(self) -> np.array:
        """
        Reset state to a random step between state space min and max
        :return: The state after reset was performed.
        """
        self.state = self.__state_step * np.random.randint(self.num_steps)
        return np.array([self.state])

    def reward(self,
               state: float) -> float:
        """
        Compute the reward for the given state;
        If x is a value in radians and state space is in range (0 to m)then the reward function is:
           sin(x)* EXP(1-ABS((x-(m/2))/(m/2)))
        :param state:
        :return: Reward for given state
        """
        x = state * self.__x_step
        return math.sin(x) * math.exp(1 - math.fabs((x - (self.__x_max / 2.0)) / (self.__x_max / 2.0)))

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
            raise RuntimeError("Action can only be value 0 or 1 so [" + str(actn) + "] is illegal")

        dn = (self.state < self.__state_min or self.state > self.__state_max)

        return np.array([self.state]), self.reward(self.state), dn

    @classmethod
    def state_space_size(cls) -> int:
        """
        The dimensions of the state space
        :return: Always 1 as this is for 1D reward functions.
        """
        return super(LocalMaximaRewardFunction1D, cls).state_space_size()

    @classmethod
    def num_actions(cls) -> int:
        """
        The number of actions
        :return: Always 2 as this is a 1D state space so only 2 directions of state space traversal.
        """
        return super(LocalMaximaRewardFunction1D, cls).num_actions()

    def state_min(self):
        """
        What is the minimum value of 1D state space
        :return: Minimum value of 1D state space
        """
        return self.__state_min

    def state_max(self):
        """
        What is the maximum value of 1D state space
        :return: Maximum value of 1D state space
        """
        return self.__state_max

    def state_step(self):
        """
        What is the discrete step increment used to traverse state space (by actions)
        :return: The discrete step increment used to traverse state space (by actions)
        """
        return self.__state_step
