import numpy as np

from reflrn.Interface.State import State


class TestState(State):

    #
    # Constructor has no arguments as it just sets the game
    #
    def __init__(self, st: np.ndarray):
        self.__st = np.array_str(st, copy=True)  # State must be immutable

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
    def state_as_array(self):
        return self.state()
