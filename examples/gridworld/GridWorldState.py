import numpy as np

from reflrn.Interface.Agent import Agent
from reflrn.Interface.State import State
from .Grid import Grid


class GridWorldState(State):

    #
    # Constructor has no arguments as it just sets the game
    #
    def __init__(self, grid: Grid):
        self.__grid = grid.deep_copy()  # State must be immutable

    #
    # An environment specific representation for Env. State
    #
    def state(self) -> object:
        return self.__grid.state()

    #
    # An string representation of the environment state
    #
    def state_as_string(self) -> str:
        st = ""
        nst = self.__grid.state()
        for cell in np.reshape(nst, len(nst)):
            if len(st) > 0:
                st = st + ","
            if np.isnan(cell):
                st += "?"
            else:
                st += str(int(cell))
        return st

    #
    # Render the board as human readable with q values adjacent if supplied
    #
    def state_as_visualisation(self) -> str:
        return self.state_as_string()

    #
    # Return the array encoded form of the grid to be used as the X input to a NN.
    #
    def state_as_array(self):
        return self.state()  # Grid-world state is an array of float
