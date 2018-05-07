from copy import deepcopy

import numpy as np

from .Grid import Grid
from .GridBlockedActionException import GridBlockedActionException
from .GridEpisodeOverException import GridEpisodeOverException


#
# Simple (small grid) with the basic North, South, East, West moves (no diagonal moves)
# The grid map of rewards is defined at construction time.
#
#


class SimpleGridOne(Grid):
    STEP = np.float(-0.2)
    FIRE = np.float(-1)
    GOAL = np.float(+1)
    BLCK = np.float(-1.1)
    FIN = GOAL
    FREE = np.float(0)
    NORTH = np.int(0)
    SOUTH = np.int(1)
    EAST = np.int(2)
    WEST = np.int(3)
    __actions = {NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1)}  # N, S ,E, W (row-offset, col-offset)

    #
    # C-Tor.
    #
    # Ensure that deep copies are taken of the supplied arguments to ensure the Grid is independent and
    # immutable.
    #
    def __init__(self,
                 grid_id: int,
                 grid_map: [],
                 start_coords: []
                 ):
        self.__grid_id = grid_id
        self.__grid = grid_map[:]  # Deep Copy
        self.__grid_rows = len(self.__grid)
        self.__grid_cols = len(self.__grid[0])
        if start_coords is not None:
            self.__start = deepcopy(start_coords)
            self.__curr = [self.__start[0], self.__start[1]]
        else:
            self.__start = deepcopy([0, 0])
            self.__curr = [0, 0]
        self.__trace = None

    #
    # What is teh shape of the grid
    #
    def shape(self) -> [int]:
        return [self.__grid_rows, self.__grid_cols]

    #
    # What is the "state" of the grid. Where state is in the context of State-Action-Reward.
    # for these simple grids the grid itself is immutable so the defined state is simply the
    # current "location" of the grid, i.e. the active cell location where the agent is.
    #
    def state(self) -> [np.float]:
        return [np.float(self.__curr[0]), np.float(self.__curr[1])]

    #
    # Reset the grid state after episode end.
    #
    def reset(self):
        self.__curr = deepcopy(self.__start)
        return

    #
    # Execute the given action and return the reward earned for moving to the new grid location.
    #
    def execute_action(self, action: int) -> np.float:
        if self.__episode_over():
            raise GridEpisodeOverException("Episode already complete, agent at finish cell on grid")
        if action not in self.allowable_actions():
            raise GridBlockedActionException("Illegal Grid Move, cell blocked or action would move out of grid")

        self.__curr = self.__new_coords_after_action(action)
        if self.__episode_over():
            self.__episode_reset()
        self.__track_location(self.__curr)
        return self.__grid_reward(self.__curr)

    #
    # What is the list of all possible actions.
    #
    def actions(self) -> [int]:
        return [self.NORTH, self.SOUTH, self.EAST, self.WEST]

    #
    # Return the given id of the grid.
    #
    def id(self) -> int:
        return self.__grid_id

    #
    # Return a deep copy of self.
    #
    def deep_copy(self) -> Grid:
        cp = type(self)(self.id(),
                        self.__grid,
                        self.__start)
        cp.__curr = self.__curr
        return cp

    #
    # Is the episode complete.
    #
    def episode_complete(self) -> bool:
        return self.__episode_over()

    #
    # Is the current location at a defined terminal (finish) location.
    #
    def __episode_over(self) -> bool:
        return self.__grid_reward(self.__curr) == self.FIN

    #
    # What will the new current coordinates be if the given action
    # was applied.
    #
    def __new_coords_after_action(self, action: int) -> [int]:
        mv = self.__actions[action]
        nw = deepcopy(self.__curr)
        nw[0] += mv[0]
        nw[1] += mv[1]
        return nw

    @classmethod
    def coords_after_action(cls, x: int, y: int, action: int) -> [int]:
        mv = cls.__actions[action]
        return [x + mv[0], y + mv[1]]

    #
    # What is the defined reward for the given grid location.
    #
    def __grid_reward(self, coords: ()) -> np.float:
        return self.__grid[coords[0]][coords[1]]

    #
    # Is the given grid location defined as blocked ?
    #
    def __blocked(self, coords: ()) -> bool:
        return self.__grid[coords[0]][coords[1]] == self.BLCK

    #
    # List of allowable actions from the current position
    #
    def allowable_actions(self) -> [int]:
        ams = []
        if self.__episode_over():
            return []
        new_coords = self.__new_coords_after_action(self.NORTH)
        if self.__curr[0] > 0 and not self.__blocked(new_coords) and not self.__location_already_visited(new_coords):
            ams.append(self.NORTH)
        new_coords = self.__new_coords_after_action(self.SOUTH)
        if self.__curr[0] + 1 < self.__grid_rows and not self.__blocked(
                new_coords) and not self.__location_already_visited(new_coords):
            ams.append(self.SOUTH)
        new_coords = self.__new_coords_after_action(self.WEST)
        if self.__curr[1] > 0 and not self.__blocked(new_coords) and not self.__location_already_visited(new_coords):
            ams.append(self.WEST)
        new_coords = self.__new_coords_after_action(self.EAST)
        if self.__curr[1] + 1 < self.__grid_cols and not self.__blocked(
                new_coords) and not self.__location_already_visited(new_coords):
            ams.append(self.EAST)
        return ams

    #
    # Reset at end of episode.
    #
    def __episode_reset(self) -> None:
        self.__trace = dict()
        return

    #
    # Has this location been already visited in the current episode
    #
    def __location_already_visited(self, coords: ()) -> bool:
        if self.__trace is not None:
            return coords[0] == self.__curr[0] and coords[1] == self.__curr[1]
        return False

    #
    # Tracxk the fact this location has been visited in current episode.
    #
    def __track_location(self, coords: ()) -> None:
        self.__trace = coords
        return
