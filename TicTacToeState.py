import numpy as np
from State import State


class TicTacToeState(State):

    #
    # Constructor has no arguments as it just sets the game
    #
    def __init__(self, board: np.array):
        self.__board = np.copy(board)  # State must be immutable

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
