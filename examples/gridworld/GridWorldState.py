import numpy as np

from reflrn import Agent
from reflrn import State
from .Grid import Grid


class GridWorldState(State):

    #
    # Constructor has no arguments as it just sets the game
    #
    def __init__(self, grid: Grid, agent_x: Agent):
        self.__grid = np.copy(board)  # State must be immutable
        self.__x_id = agent_x.id()
        self.__x_name = agent_x.name()

    #
    # An environment specific representation for Env. State
    #
    def state(self) -> object:
        return None

    #
    # An string representation of the environment state
    #
    def state_as_string(self) -> str:
        st = ""
        for cell in np.reshape(self.__board, self.__board.size):
            if np.isnan(cell):
                st += "0"
            else:
                st += str(int(cell))
        return st

    #
    # Render the board as human readable with q values adjacent if supplied
    #
    def state_as_visualisation(self) -> str:
        s = ""
        for i in range(0, 3):
            rbd = ""
            for j in range(0, 3):
                rbd += "["
                if np.isnan(self.__board[i][j]):
                    rbd += " "
                else:
                    rbd += self.__agents[self.__board[i][j]]
                rbd += "]"
            s += rbd + "\n"
        s += "\n"
        return s
