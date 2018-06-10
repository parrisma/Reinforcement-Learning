import abc

import numpy as np

from reflrn.Interface.State import State


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
    def execute_action(self, action: int) -> int:  # ToDo is this type hint correct ?
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
    # Convert the allowable actions into a boolean mask.
    #
    @abc.abstractmethod
    def disallowed_actions(self, allowable_actions) -> [int]:
        pass

    @classmethod
    @abc.abstractmethod
    def coords_after_action(cls, x: int, y: int, action: int) -> [int]:
        pass

    #
    # What is the reward for the given grid location.
    #
    @abc.abstractmethod
    def reward(self, x: int, y: int) -> np.float:
        pass

    #
    # Reset the grid to its state as at construction.
    #
    @abc.abstractmethod
    def reset(self):
        pass

    #
    # Is the episode complete, are we at the terminal finish state ? If the coords are supplied
    # then test those coords rather than the current grid location.
    #
    @abc.abstractmethod
    def episode_complete(self, coords: tuple = None) -> bool:
        pass

    #
    # Return a State representation of Grid World.
    #
    @abc.abstractmethod
    def state(self) -> State:
        pass

    #
    # Return grid dimensions
    #
    @abc.abstractmethod
    def shape(self) -> [int]:
        pass
