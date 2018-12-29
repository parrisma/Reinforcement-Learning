import numpy as np

from reflrn.Interface.Agent import Agent
from reflrn.Interface.State import State


class TicTacToeState(State):

    #
    # Constructor.
    #
    def __init__(self,
                 board: np.array,
                 agent_x: Agent,
                 agent_o: Agent):
        self.__board = np.copy(board)  # State must be immutable
        self.__x_id = agent_x.id()
        self.__x_name = agent_x.name()
        self.__o_id = agent_o.id()
        self.__o_name = agent_o.name()
        self.__agents = dict()
        self.__agents[self.__x_id] = self.__x_name
        self.__agents[self.__o_id] = self.__o_name
        self.__unused = self.__o_id + self.__x_id
        self.__agent_x = agent_x
        self.__agent_o = agent_o

    #
    # An environment specific representation for Env. State
    #
    def state(self) -> object:
        return np.copy(self.__board)

    #
    # Return the id of the other agent
    #
    def __other(self,
                agent_id: int) -> int:
        if agent_id == self.__x_id:
            return self.__o_id
        elif agent_id == self.__o_id:
            return self.__x_id
        return agent_id

    #
    # Return a new state with an invert the player perspective of the board.
    #
    def invert_player_perspective(self) -> State:
        brd = np.copy(self.__board)
        shp = brd.shape
        brd = np.reshape(brd, np.size(brd))
        brd = np.array([self.__other(x) for x in brd])
        brd = np.reshape(brd, shp)
        return TicTacToeState(board=brd,
                              agent_x=self.__agent_x,
                              agent_o=self.__agent_o)

    #
    # An string representation of the environment curr_coords
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

    #
    # State encoded as a numpy array that can be passed as the X (input) into
    # a Neural Net. The dimensionality can vary depending on the implementation
    # from a linear vector for a simple Sequential model to an 3D array for a
    # multi layer convolutional model.
    #
    def state_as_array(self) -> np.ndarray:
        bc = np.copy(self.__board)
        bc[np.isnan(bc)] = self.__unused
        return bc
