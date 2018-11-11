from typing import Tuple

import numpy as np

from reflrn.Interface.State import State


class TestState(State):

    #
    # Constructor takes a numpy array which is the state of the environment.
    #
    def __init__(self,
                 st: np.ndarray,
                 shp: Tuple = None):
        self.__st = np.array(st, copy=True)  # State must be immutable
        if shp is not None:
            self.__st = np.reshape(self.__st, (1, np.size(self.__st)))
            self.__st = np.reshape(self.__st, shp)
        return

    #
    # An environment specific representation for Env. State
    #
    def state(self) -> np.ndarray:
        return self.__st

    #
    # An string representation of the environment curr_coords
    #
    def state_as_string(self) -> str:
        return np.array_str(self.__st)

    #
    # Render the board as human readable with q values adjacent if supplied
    #
    def state_as_visualisation(self) -> str:
        return self.state_as_string()

    #
    # Return the array encoded form of the grid to be used as the X input to a NN.
    #
    # This is always (1, n)
    #
    def state_as_array(self):
        return np.reshape(self.__st, (1, np.size(self.__st)))
