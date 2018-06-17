from copy import deepcopy
from random import randint, choice
from typing import List

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
    FIRE = np.float(-0.99)
    GOAL = np.float(+0.99)
    BLCK = np.float(-0.999)
    DISA = np.float(-1.0)
    FIN = GOAL
    FREE = np.float(0)
    NORTH = np.int(0)
    SOUTH = np.int(1)
    EAST = np.int(2)
    WEST = np.int(3)
    __actions = {NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1)}  # N, S ,E, W (row-offset, col-offset)
    RESPAWN_RANDOM = 0
    RESPAWN_CORNER = 1
    RESPAWN_EDGE = 2
    RESPAWN_DEFAULT = 3

    #
    # C-Tor.
    #
    # Ensure that deep copies are taken of the supplied arguments to ensure the Grid is independent and
    # immutable.
    #
    def __init__(self,
                 grid_id: int,
                 grid_map: [],
                 st_coords: List[int] = None,
                 respawn_type=RESPAWN_RANDOM
                 ):
        self.__grid_id = grid_id
        self.__num_actions = 4  # N,S,E,W
        self.__grid = grid_map[:]  # Deep Copy
        self.__grid_rows = len(self.__grid)
        self.__grid_cols = len(self.__grid[0])
        self.__st_coords = st_coords
        self.__last_coords = None
        self.__respawn_type = respawn_type
        self.__corners = None
        self.__edges = None
        self.__respawn_operator = {
            SimpleGridOne.RESPAWN_DEFAULT: self.__respawn_default,
            SimpleGridOne.RESPAWN_CORNER: self.__respawn_on_a_corner,
            SimpleGridOne.RESPAWN_EDGE: self.__respawn_on_an_edge,
            SimpleGridOne.RESPAWN_RANDOM: self.__respawn_random
        }
        self.__start = None
        self.__curr = None
        if self.__st_coords is not None:
            self.__start = st_coords
            self.__curr = list([self.__start[0], self.__start[1]])
        else:
            self.__start = list([0, 0])
            self.__curr = deepcopy(self.__start)

    def start_coords(self) -> List[int]:
        return (self.__respawn_operator[self.__respawn_type])()

    #
    # Re Spawn on the single start point given to constructor or [0, 0] if
    # no start coords were given.
    #
    def __respawn_default(self) -> List[int]:
        return deepcopy(self.__start)

    #
    # Re-Spawn anywhere on the grid.
    #
    def __respawn_random(self) -> List[int]:
        cl = randint(0, self.__grid_cols - 1)
        rw = randint(0, self.__grid_rows - 1)
        rc = (rw, cl)
        return rc

    #
    # Re-Spawn on any corner.
    #
    def __respawn_on_a_corner(self) -> List[int]:
        if self.__corners is None:
            self.__corners = []
            self.__corners.append((0, 0))
            self.__corners.append((0, self.__grid_cols - 1))
            self.__corners.append((self.__grid_rows - 1, 0))
            self.__corners.append((self.__grid_rows - 1, self.__grid_cols - 1))
        return deepcopy(choice(self.__corners))

    #
    # Re-Spawn any where on an edge.
    #
    def __respawn_on_an_edge(self) -> List[int]:
        if self.__edges is None:
            self.__edges = []
            for cl in range(0, self.__grid_cols):
                self.__edges.append((0, cl))
                self.__edges.append((self.__grid_rows - 1, cl))
            for rw in range(1, self.__grid_rows - 1):
                self.__edges.append((rw, 0))
                self.__edges.append((rw, self.__grid_cols - 1))

        return deepcopy(choice(self.__edges))

    #
    # What is the shape of the grid
    #
    def shape(self) -> List[int]:
        return [self.__grid_rows, self.__grid_cols]

    #
    # What is the "curr_coords" of the grid. Where curr_coords is in the context of State-Action-Reward.
    # for these simple grids the grid itself is immutable so the defined curr_coords is simply the
    # current "location" of the grid, i.e. the active cell location where the agent is.
    #
    def curr_coords(self) -> List[int]:
        return deepcopy(self.__curr)

    #
    # Return the last coords before current curr_coords
    #
    def last_coords(self) -> List[int]:
        return deepcopy(self.__last_coords)

    #
    # Reset the grid curr_coords after episode end.
    #
    def reset(self,
              coords: List[int] = None) -> None:
        if coords is None:
            self.__curr = self.start_coords()
        else:
            self.__curr = [coords[self.ROW], coords[self.COL]]
        return

    #
    # Execute the given action and return the reward earned for moving to the new grid location.
    #
    def execute_action(self, action: int) -> np.float:
        if self.__episode_over():
            raise GridEpisodeOverException("Episode already complete, agent at finish cell on grid")
        if action not in self.allowable_actions():
            raise GridBlockedActionException("Illegal Grid Move, cell blocked or action would move out of grid")

        self.__track_last_coords(self.__curr)
        self.__curr = self.__new_coords_after_action(action)
        if self.__episode_over():
            self.__episode_reset()
        return self.__grid_reward(self.__curr)

    #
    # What is the reward for the given grid location.
    #
    def reward(self, rw: int, cl: int) -> np.float:
        return self.__grid_reward([rw, cl])

    #
    # What is the list of all possible actions.
    #
    def actions(self) -> List[int]:
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
    def episode_complete(self,
                         coords: List[int] = None) -> bool:
        if coords is not None:
            return self.__episode_over(coords)
        else:
            return self.__episode_over(self.__curr)

    #
    # Is the current location at a defined terminal (finish) location.
    #
    def __episode_over(self,
                       coords: List[int] = None) -> bool:
        if coords is None:
            nw = deepcopy(self.__curr)
        else:
            nw = coords
        return self.__grid_reward(nw) == self.FIN

    #
    # What will the new current coordinates be if the given action
    # was applied.
    #
    def __new_coords_after_action(self,
                                  action: int,
                                  coords: List[int] = None) -> List[int]:
        mv = self.__actions[action]
        if coords is None:
            nw = deepcopy(self.__curr)
        else:
            nw = coords
        return list((nw[self.ROW] + mv[self.ROW], nw[self.COL] + mv[self.COL]))

    #
    # What *would* the coordinates be if the given action were to be executed
    # from the given grid location.
    #
    @classmethod
    def coords_after_action(cls, rw: int, cl: int, action: int) -> List[int]:
        mv = cls.__actions[action]
        return list((rw + mv[cls.ROW], cl + mv[cls.COL]))

    #
    # What is the defined reward for the given grid location.
    #
    def __grid_reward(self, coords: List[int]) -> np.float:
        return self.__grid[coords[self.ROW]][coords[self.COL]]

    #
    # Is the given grid location defined as blocked ?
    #
    def __blocked(self, coords: List[int]) -> bool:
        return self.__grid[coords[self.ROW]][coords[self.COL]] == self.BLCK

    #
    # Convert the allowable actions into a boolean mask.
    #
    def disallowed_actions(self, allowable_actions: List[int]) -> List[int]:
        da = []
        for i in range(0, self.__num_actions):
            if i not in allowable_actions:
                da.append(i)
        return da

    #
    # List of allowable actions from the current position
    #
    def allowable_actions(self,
                          origin: List[int] = None) -> List[int]:
        ams = list()
        if origin is None:
            coords = deepcopy(self.__curr)
        else:
            coords = deepcopy(origin)
        if self.__episode_over(coords):
            return []
        new_coords = self.__new_coords_after_action(self.NORTH, coords)
        if new_coords[self.ROW] >= 0 and not self.__blocked(new_coords):
            ams.append(self.NORTH)
        new_coords = self.__new_coords_after_action(self.SOUTH, coords)
        if new_coords[self.ROW] < self.__grid_rows and not self.__blocked(new_coords):
            ams.append(self.SOUTH)
        new_coords = self.__new_coords_after_action(self.WEST, coords)
        if new_coords[self.COL] >= 0 and not self.__blocked(new_coords):
            ams.append(self.WEST)
        new_coords = self.__new_coords_after_action(self.EAST, coords)
        if new_coords[self.COL] < self.__grid_cols and not self.__blocked(new_coords):
            ams.append(self.EAST)
        return ams

    #
    # Reset at end of episode.
    #
    def __episode_reset(self) -> None:
        self.__last_coords = None
        return

    #
    # Track the fact this location has been visited in current episode.
    #
    def __track_last_coords(self, coords: List[int]) -> None:
        self.__last_coords = coords
        return
