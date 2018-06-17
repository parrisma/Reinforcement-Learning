import abc
from typing import List

import numpy as np


#
# This is the interface that all Grid World Grids need to implement.
#
class Grid(metaclass=abc.ABCMeta):
    ROW = 0
    COL = 1

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
    def actions(self) -> List[int]:
        pass

    #
    # Execute the given (allowable) action and return the reward and the curr_coords
    #
    @abc.abstractmethod
    def execute_action(self, action: int) -> np.float:
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
    def allowable_actions(self,
                          coords: List[int] = None) -> List[int]:
        pass

    #
    # Convert the allowable actions into a boolean mask.
    #
    @abc.abstractmethod
    def disallowed_actions(self, allowable_actions: List[int]) -> List[int]:
        pass

    @classmethod
    @abc.abstractmethod
    def coords_after_action(cls, x: int, y: int, action: int) -> List[int]:
        pass

    #
    # What is the reward for the given grid location.
    #
    @abc.abstractmethod
    def reward(self, x: int, y: int) -> np.float:
        pass

    #
    # Reset the grid to its curr_coords as at construction.
    #
    @abc.abstractmethod
    def reset(self):
        pass

    #
    # Is the episode complete, are we at the terminal finish curr_coords ? If the coords are supplied
    # then test those coords rather than the current grid location.
    #
    @abc.abstractmethod
    def episode_complete(self, coords: List[int] = None) -> bool:
        pass

    #
    # Return a current coordinates of Grid World.
    #
    @abc.abstractmethod
    def curr_coords(self) -> List[int]:
        pass

    #
    # Return a last curr_coords of Grid World.
    #
    @abc.abstractmethod
    def last_coords(self) -> List[int]:
        pass

    #
    # Return grid dimensions
    #
    @abc.abstractmethod
    def shape(self) -> List[int]:
        pass
