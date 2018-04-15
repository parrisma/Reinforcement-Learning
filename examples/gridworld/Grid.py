import abc
import numpy as np


#
# This is the interface that all Grid World Grids need to implement.
#
class Grid(metaclass=abc.ABCMeta):

    #
    # Immutable Id for the Grid
    #
    @abc.abstractmethod
    def id(self) -> int:
        pass

    #
    # Return the list of possible actions
    #
    @abc.abstractmethod
    def actions(self) -> [int]:
        pass

    #
    # Execute the given (allowable) action and return the reward and the state
    #
    @abc.abstractmethod
    def execute_action(self, action: int) -> int:
        pass

    #
    # Deep Copy the Grid.
    #
    @abc.abstractmethod
    def deep_copy(self):
        pass

    #
    # What actions are allowable with agent at current location.
    #
    @abc.abstractmethod
    def allowable_actions(self) -> [int]:
        pass

    #
    # Reset the grid to its state as at construction.
    #
    @abc.abstractmethod
    def reset(self):
        pass

    #
    # Is the episode complete, are we at the terminal finish state ?
    #
    @abc.abstractmethod
    def episode_complete(self) -> bool:
        pass

    #
    # Return a numpy array of the grid "state"
    #
    @abc.abstractmethod
    def state(self) -> [np.float]:
        pass
