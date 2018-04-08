from .Grid import Grid

#
# Simple (small grid) with the basic North, South, East, West moves (no diagonal moves)
#
# -1 penalty for every move => push to find shortest path
# F = Finish
# S = Start
# XX = Blocked
#
# [  -1][ -100][F+100][  -1][  -1]
# [  -1][  XX ][ XX  ][  -1][  -1]
# [  -1][  XX ][ XX  ][ XX ][  -1]
# [S -1][   -1][   -1][  -1][  -1]
#


class SimpleGridOne(Grid):

    __start = [0, 3]
    __finish = [2, 0]
    __move = -1
    __fire = -100
    __goal = +100
    __blck = -101
    __action = dict()
    __actions = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}  # N, S ,E, W (row-offset, col-offset)
    __grid = [
                [__move, __fire, __goal, __move, __move],
                [__move, __blck, __blck, __move, __move],
                [__move, __blck, __blck, __blck, __move],
                [__move, __move, __move, __move, __move]
    ]

    def __init__(self, id: int):
        self.__id = id
        self.__curr = [self.__start[0], self.__start[1]]

    def execute_action(self, action: int) -> int:
        pass

    def actions(self) -> [int]:
        return self.__actions.copy()

    def id(self) -> int:
        return self.__id

    def copy(self) -> Grid:
        cp = SimpleGridOne(self.id())
