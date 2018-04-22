import numpy as np

from reflrn.Interface.Agent import Agent
from reflrn.Interface.State import State
from .Grid import Grid


class GridWorldState(State):

    #
    # Constructor has no arguments as it just sets the game
    #
    def __init__(self, grid: Grid, agent_x: Agent):
        self.__grid = grid.deep_copy()  # State must be immutable
        self.__x_id = agent_x.id()
        self.__x_name = agent_x.name()

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
