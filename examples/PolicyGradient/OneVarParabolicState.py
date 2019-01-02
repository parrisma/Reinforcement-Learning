import numpy as np

from reflrn.Interface.State import State


#
# The state is just the current value of 'x'
#

class OneVarParabolicState(State):

    def __init__(self,
                 x: float):
        self.__state = np.array([x])

    def state(self) -> object:
        return np.copy(self.__state)  # State is immutable

    def state_as_string(self) -> str:
        return np.array2string(self.__state, separator=',')

    def state_as_array(self) -> np.ndarray:
        return np.copy(self.__state)  # State is immutable
