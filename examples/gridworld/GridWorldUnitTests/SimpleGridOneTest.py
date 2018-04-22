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
# Execute the ReflrnUnitTests.
#


if __name__ == "__main__":
    if True:
        tests = TestSimpleGridOne()
        suite = unittest.TestLoader().loadTestsFromModule(tests)
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(TestSimpleGridOne("test_2"))
        unittest.TextTestRunner().run(suite)
