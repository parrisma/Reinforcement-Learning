import numpy as np

from examples.gridworld.Grid import Grid
from examples.gridworld.SimpleGridOne import SimpleGridOne


class GridFactory:
    step = SimpleGridOne.STEP
    fire = SimpleGridOne.FIRE
    blck = SimpleGridOne.BLCK
    goal = SimpleGridOne.GOAL

    @classmethod
    def test_grid_one(cls) -> Grid:
        grid = [
            [cls.step, cls.fire, cls.goal, cls.step, cls.step],
            [cls.step, cls.blck, cls.blck, cls.fire, cls.step],
            [cls.step, cls.blck, cls.blck, cls.blck, cls.step],
            [cls.step, cls.step, cls.step, cls.step, cls.step]
        ]

        sg1 = SimpleGridOne(3,
                            grid,
                            [3, 0])

        return sg1

    @classmethod
    def test_grid_two(cls) -> Grid:
        grid = [
            [cls.step, cls.step, cls.step, cls.step, cls.goal]
        ]

        sg1 = SimpleGridOne(3,
                            grid,
                            [0, 0])

        return sg1

    @classmethod
    def test_grid_three(cls) -> Grid:
        grid = [
            [cls.step, cls.step, cls.step, cls.step, cls.goal],
            [cls.step, cls.step, cls.step, cls.step, cls.step],
            [cls.step, cls.step, cls.step, cls.step, cls.step],
            [cls.step, cls.step, cls.step, cls.step, cls.step],
            [cls.step, cls.step, cls.step, cls.step, cls.step]
        ]

        sg1 = SimpleGridOne(3,
                            grid,
                            [4, 0])
        return sg1

    @classmethod
    def test_grid_four(cls) -> Grid:
        r = 20
        c = 20
        grid = np.full((r, c), cls.step)
        grid[5][5] = cls.goal
        grid[15][15] = cls.fire
        sg1 = SimpleGridOne(4,
                            grid,
                            [r-1, c-1])
        return sg1
