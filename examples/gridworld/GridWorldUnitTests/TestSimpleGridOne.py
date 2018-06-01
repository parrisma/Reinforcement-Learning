import unittest

from examples.gridworld.GridBlockedActionException import GridBlockedActionException
from examples.gridworld.GridEpisodeOverException import GridEpisodeOverException
from examples.gridworld.SimpleGridOne import SimpleGridOne


class TestSimpleGridOne(unittest.TestCase):
    step = SimpleGridOne.STEP
    fire = SimpleGridOne.FIRE
    blck = SimpleGridOne.BLCK
    goal = SimpleGridOne.GOAL

    #
    # Test, all possible moves on 3 by 3 grid
    #
    def test_0(self):
        grid = [
            [self.step, self.step, self.step],
            [self.step, self.step, self.step],
            [self.step, self.step, self.step]
        ]
        sg4 = SimpleGridOne(0,
                            grid,
                            [1, 1])

        test_cases = [[(0, 0), 2, [SimpleGridOne.SOUTH, SimpleGridOne.EAST]],
                      [(0, 1), 3, [SimpleGridOne.WEST, SimpleGridOne.SOUTH, SimpleGridOne.EAST]],
                      [(0, 2), 2, [SimpleGridOne.WEST, SimpleGridOne.SOUTH]],
                      [(1, 0), 3, [SimpleGridOne.NORTH, SimpleGridOne.EAST, SimpleGridOne.SOUTH]],
                      [(1, 1), 4, [SimpleGridOne.NORTH, SimpleGridOne.EAST, SimpleGridOne.SOUTH, SimpleGridOne.EAST]],
                      [(1, 2), 3, [SimpleGridOne.WEST, SimpleGridOne.NORTH, SimpleGridOne.SOUTH]],
                      [(2, 0), 2, [SimpleGridOne.NORTH, SimpleGridOne.EAST]],
                      [(2, 1), 3, [SimpleGridOne.NORTH, SimpleGridOne.WEST, SimpleGridOne.EAST]],
                      [(2, 2), 2, [SimpleGridOne.WEST, SimpleGridOne.NORTH]]
                      ]

        for coords, ln, moves in test_cases:
            aac = sg4.allowable_actions(coords)
            self.assertEqual(len(aac), ln)
            for mv in moves:
                self.assertTrue(mv in aac)

        return

    #
    # Test
    #
    def test_1(self):
        grid = [
            [self.goal]
        ]
        sg1 = SimpleGridOne(1,
                            grid,
                            [0, 0])
        alm = sg1.allowable_actions()
        self.assertEqual(alm, [])
        for actn in sg1.actions():
            self.assertRaises(GridEpisodeOverException, sg1.execute_action, actn)  # Over as start == finish
        return

    #
    # Test
    #
    def test_2(self):
        grid = [
            [self.goal, self.step],
            [self.step, self.fire]
        ]
        sg1 = SimpleGridOne(2,
                            grid,
                            [1, 0])

        # At start, can go North & West
        alm = sg1.allowable_actions()
        self.assertEqual(len(alm), 2)
        self.assertEqual(SimpleGridOne.NORTH in alm, True)
        self.assertEqual(SimpleGridOne.EAST in alm, True)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.SOUTH)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.WEST)

        # Go East
        rw = sg1.execute_action(SimpleGridOne.EAST)
        self.assertEqual(rw, self.fire)
        alm = sg1.allowable_actions()
        self.assertEqual(len(alm), 2)
        self.assertEqual(SimpleGridOne.NORTH in alm, True)
        self.assertEqual(SimpleGridOne.WEST in alm, True)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.SOUTH)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.EAST)

        # Go North
        rw = sg1.execute_action(SimpleGridOne.NORTH)
        self.assertEqual(rw, self.step)
        alm = sg1.allowable_actions()
        self.assertEqual(len(alm), 2)
        self.assertEqual(SimpleGridOne.SOUTH in alm, True)
        self.assertEqual(SimpleGridOne.WEST in alm, True)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.NORTH)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.EAST)

        # Go West (Done)
        rw = sg1.execute_action(SimpleGridOne.WEST)
        self.assertEqual(rw, self.goal)
        alm = sg1.allowable_actions()
        self.assertEqual(alm, [])
        for actn in sg1.actions():
            self.assertRaises(GridEpisodeOverException, sg1.execute_action, actn)

        return

    def test_3(self):
        grid = [
            [self.step, self.fire, self.goal, self.step, self.step],
            [self.step, self.blck, self.blck, self.fire, self.step],
            [self.step, self.blck, self.blck, self.blck, self.step],
            [self.step, self.step, self.step, self.step, self.step]
        ]
        sg1 = SimpleGridOne(3,
                            grid,
                            [3, 0])

        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.SOUTH)
        self.assertEqual(sg1.execute_action(SimpleGridOne.EAST), self.step)
        self.assertEqual(sg1.execute_action(SimpleGridOne.EAST), self.step)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.NORTH)
        self.assertEqual(sg1.execute_action(SimpleGridOne.EAST), self.step)
        self.assertEqual(sg1.execute_action(SimpleGridOne.EAST), self.step)
        self.assertEqual(sg1.execute_action(SimpleGridOne.NORTH), self.step)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.WEST)
        self.assertEqual(sg1.execute_action(SimpleGridOne.NORTH), self.step)
        self.assertEqual(sg1.execute_action(SimpleGridOne.WEST), self.fire)
        self.assertEqual(sg1.execute_action(SimpleGridOne.NORTH), self.step)
        self.assertRaises(GridBlockedActionException, sg1.execute_action, SimpleGridOne.NORTH)
        self.assertEqual(sg1.execute_action(SimpleGridOne.WEST), self.goal)
        self.assertRaises(GridEpisodeOverException, sg1.execute_action, SimpleGridOne.WEST)

        return

    #
    # Test, where current coords are passed rather than taken from the internal state of the grid.
    #
    def test_4(self):
        grid = [
            [self.step, self.step, self.step],
            [self.goal, self.step, self.step],
            [self.step, self.fire, self.step],
            [self.step, self.step, self.blck],
            [self.step, self.step, self.step]
        ]
        sg4 = SimpleGridOne(4,
                            grid,
                            [4, 2])

        self.assertEqual(sg4.episode_complete(), False)
        coords = (1, 0)
        self.assertEqual(sg4.episode_complete(coords), True)

        coords = (4, 2)
        aa = sg4.allowable_actions()
        aac = sg4.allowable_actions(coords)
        self.assertTrue(SimpleGridOne.WEST in aac)
        self.assertTrue(len(aac), 1)
        self.assertTrue(aa == aac)

        coords = (0, 0)
        aac = sg4.allowable_actions(coords)
        self.assertTrue(SimpleGridOne.EAST in aac)
        self.assertTrue(SimpleGridOne.SOUTH in aac)
        self.assertTrue(len(aac), 2)

        return


#
# Execute the UnitTests.
#


if __name__ == "__main__":
    if True:
        tests = TestSimpleGridOne()
        suite = unittest.TestLoader().loadTestsFromModule(tests)
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(TestSimpleGridOne("test_0"))
        unittest.TextTestRunner().run(suite)
