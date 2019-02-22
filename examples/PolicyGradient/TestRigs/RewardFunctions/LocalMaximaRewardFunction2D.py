import math
from threading import Thread
from time import sleep
from typing import Tuple

import numpy as np

from examples.PolicyGradient.TestRigs.Interface.RewardFunction2D import RewardFunction2D

"""
In 3D Space (but 2D in the sense 2 degrees of freedom in state space) this Reward Function has a local maximas 
as aa series of small peaks around a global maxima peak.
 
This is modelled as 2.5 cycles of a sinusoidal curve with the 2nd (central) peak weighed as give it a larger 
magnitude than the peeks either side. The function is symmetrical around the turning point of the central peak.

So the RL Agent should be able to find and maximise to move the agent to the central peak even
when starting at the state space extremity and having to pass through the local maxima. 

If x and y is a value in radians and state space is in range (0 to m)then the reward function is:
   sin(x)* EXP(1-ABS((x-(m/2))/(m/2)))

"""


class LocalMaximaRewardFunction2D(RewardFunction2D):
    __state_min = int(0)
    __state_max = int(60)
    __state_step = int(1)
    __center_state = int(__state_max / 2.0)
    __x_min = float(0)
    __x_max = float(2.5 * (2 * math.pi))
    __x_step = float(15.0 * (math.pi / 180.0))  # 15 degree steps as radians
    __y_min = __x_min
    __y_max = __x_max
    __y_step = __x_step

    def __init__(self):
        """
        Start in a default reset state.
        """
        self.state = None
        self.num_steps = int((self.__state_max - self.__state_min) / self.__state_step)
        self.done_state = int(self.num_steps / 2)
        self.reset()
        return

    def reset(self) -> np.array:
        """
        Reset {x, y] state to a random step between state space min and max
        :return: Random x, y (within state space) as numpy array
        """
        self.state = [self.__state_step * np.random.randint(self.num_steps),
                      self.__state_step * np.random.randint(self.num_steps)]
        return np.array([self.state[0], self.state[1]])

    def reward(self,
               state: Tuple[int, int]) -> float:
        """
        Compute the reward for the given state;
        If state (x-pos, y-pos) are values as integers in range (0 to m) where m = (x_max - x_min) / x_step
        converted to radians as x = x-pos * x-step, then the reward is
           r = sin(x)* EXP(1-ABS((x-(m/2))/(m/2))) + sin(y)* EXP(1-ABS((y-(m/2))/(m/2)))
        :param: state: as Tuple[x-pos: int, y-pos:int]
        :return: Reward for given state (x-pos, y-pos)
        """
        x = state[0] * self.__x_step
        y = state[1] * self.__y_step
        xcomp = math.sin(x) * math.exp(1 - math.fabs((x - (self.__x_max / 2.0)) / (self.__x_max / 2.0)))
        ycomp = math.sin(y) * math.exp(1 - math.fabs((y - (self.__x_max / 2.0)) / (self.__x_max / 2.0)))
        return xcomp + ycomp

    def step(self,
             actn: int) -> Tuple[np.array, float, bool]:
        """
        Take the specified action
        :param actn: the action to take
        :return: The new state, the reward for the state transition and bool, which is true if episode ended
        """
        if actn == 0:  # N
            self.state[1] += self.__state_step  # y+
        elif actn == 1:  # E
            self.state[0] += self.__state_step  # X+
        elif actn == 2:  # S
            self.state[1] -= self.__state_step  # y-
        elif actn == 3:  # W
            self.state[0] -= self.__state_step  # x-
        else:
            raise ValueError("Action can only be value 0 or 3 so [" + str(actn) + "] is illegal")

        dnx = (self.state[0] < self.__state_min or self.state[0] > self.__state_max or self.state[0] == self.done_state)
        dny = (self.state[1] < self.__state_min or self.state[1] > self.__state_max or self.state[1] == self.done_state)

        return np.array([self.state[0], self.state[1]]), self.reward(self.state), (dnx or dny)

    @classmethod
    def state_space_dimension(cls) -> int:
        """
        The dimensions of the state space
        :return: Always 1 as this is for 1D reward functions.
        """
        return super(LocalMaximaRewardFunction2D, cls).state_space_dimension()

    @classmethod
    def num_actions(cls) -> int:
        """
        The number of actions. Always 4 as this is a 2D state space so only 4 directions of state space traversal.
        North, South, East, West
        :return: Number of actions as integer
        """
        return super(LocalMaximaRewardFunction2D, cls).num_actions()

    def state_min(self) -> float:
        """
        What is the minimum value of 2D state space. Both axis are constrained to same dimensions
        :return: Minimum value of 2D state space
        """
        return self.__state_min

    def state_max(self) -> float:
        """
        What is the maximum value of 2D state space. Both axis are constrained to same dimensions
        :return: Maximum value of 2D state space
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
        return super(LocalMaximaRewardFunction2D, self).state_shape()


#
# Simple Demo
#
def do_plot():
    lm2d = LocalMaximaRewardFunction2D()
    for i in range(1, 10):
        lm2d.plot()
        sleep(1)
    return


if __name__ == "__main__":
    t = Thread(target=do_plot())
    t.daemon = True
    t.start()
    print("Done")
