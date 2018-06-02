import logging
import random
import unittest
from random import shuffle

import numpy as np

from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.RareEventBiasReplayMemory import RareEventBiasReplayMemory


class TestRareEventBiasReplayMemory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        random.seed(42)
        np.random.seed(42)
        cls.__lg = EnvironmentLogging("TestRareEventBiasReplayMemory",
                                      "TestRareEventBiasReplayMemory.log",
                                      logging.DEBUG
                                      ).get_logger()

    def test_empty_memory(self):
        rebrm = RareEventBiasReplayMemory(self.__lg)
        self.assertRaises(RareEventBiasReplayMemory.SampleMemoryTooSmall,
                          rebrm.get_random_memories,
                          1)

    def test_single_memory(self):
        rebrm_single = RareEventBiasReplayMemory(self.__lg)
        rebrm_single.append_memory("S1", "Sn", 0, 0.5, False)
        samples = rebrm_single.get_random_memories(1)
        self.assertEqual(1, len(samples))
        self.assertRaises(RareEventBiasReplayMemory.SampleMemoryTooSmall,
                          rebrm_single.get_random_memories,
                          2)
        d = self.as_dict(samples)
        self.assertTrue("S1" in d)
        return

    def test_double_memory(self):
        rebrm_single = RareEventBiasReplayMemory(self.__lg)
        rebrm_single.append_memory("S1", "S1n", 0, 0.5, False)
        rebrm_single.append_memory("S2", "S2n", 0, 0.5, False)
        samples = rebrm_single.get_random_memories(2)
        self.assertEqual(2, len(samples))
        self.assertRaises(RareEventBiasReplayMemory.SampleMemoryTooSmall,
                          rebrm_single.get_random_memories,
                          3)
        d = self.as_dict(samples)
        self.assertTrue("S1" in d)
        self.assertTrue("S2" in d)
        return

    def test_many_same_memory(self):
        rebrm_single = RareEventBiasReplayMemory(self.__lg)
        for i in range(1, 100):
            rebrm_single.append_memory("S" + str(i), "S" + str(i) + "n", 0, 0.5, False)
        samples = rebrm_single.get_random_memories(30)
        self.assertEqual(30, len(samples))
        self.assertRaises(RareEventBiasReplayMemory.SampleMemoryTooSmall,
                          rebrm_single.get_random_memories,
                          101)
        return

    def test_many_one_rare_memory(self):
        rebrm_single = RareEventBiasReplayMemory(self.__lg)
        for i in range(1, 100):
            rebrm_single.append_memory("S" + str(i), "S" + str(i) + "n", 0, 0.5, False)
        rebrm_single.append_memory("SRare1", "SRare1N" + str(i) + "n", 0, 1.0, False)
        samples = rebrm_single.get_random_memories(30)
        self.assertEqual(30, len(samples))
        self.assertRaises(RareEventBiasReplayMemory.SampleMemoryTooSmall,
                          rebrm_single.get_random_memories,
                          101)
        d = self.as_dict(samples)
        self.assertTrue("SRare1" in d)
        return

    def test_many_rare_memory(self):
        rebrm_single = RareEventBiasReplayMemory(self.__lg)
        for i in range(1, 200):
            rebrm_single.append_memory("S" + str(i), "S" + str(i) + "n", 0, 0.5, False)
        rebrm_single.append_memory("SRare1", "SRare1N" + str(i) + "n", 0, 1.0, False)
        samples = rebrm_single.get_random_memories(30)
        self.assertEqual(30, len(samples))
        d = self.as_dict(samples)
        self.assertTrue("SRare1" in d)

        rebrm_single.append_memory("SRare2", "SRare2N" + str(i) + "n", 0, -1.0, False)
        rebrm_single.append_memory("SRare3", "SRare3N" + str(i) + "n", 0, -1.0, False)
        samples = rebrm_single.get_random_memories(30)
        self.assertEqual(30, len(samples))
        d = self.as_dict(samples)
        r1 = "SRare2" in d
        r2 = "SRare3" in d
        self.assertTrue(r1 or r2)

        rebrm_single.append_memory("SRare4", "SRare4N" + str(i) + "n", 0, 1.5, False)
        rebrm_single.append_memory("SRare5", "SRare5N" + str(i) + "n", 0, 1.5, False)
        rebrm_single.append_memory("SRare6", "SRare6N" + str(i) + "n", 0, 1.5, False)
        samples = rebrm_single.get_random_memories(30)
        self.assertEqual(30, len(samples))
        d = self.as_dict(samples)
        r4 = "SRare4" in d
        r5 = "SRare5" in d
        r6 = "SRare6" in d
        self.assertTrue(r4 or r5 or r6)
        return

    def test_many__shuffled_rare_memory(self):
        test_cases = []
        for i in range(1, 200):
            test_cases.append(["S" + str(i), "S" + str(i) + "n", 0, 0.5, False])
        test_cases.append(["SRare1", "SRare1N" + str(i) + "n", 0, 1.0, False])
        test_cases.append(["SRare2", "SRare2N" + str(i) + "n", 0, -1.0, False])
        test_cases.append(["SRare3", "SRare3N" + str(i) + "n", 0, -1.0, False])
        test_cases.append(["SRare4", "SRare4N" + str(i) + "n", 0, 1.5, False])
        test_cases.append(["SRare5", "SRare5N" + str(i) + "n", 0, 1.5, False])
        test_cases.append(["SRare6", "SRare6N" + str(i) + "n", 0, 1.5, False])
        shuffle(test_cases)

        rebrm_single = RareEventBiasReplayMemory(self.__lg)
        for case in test_cases:
            rebrm_single.append_memory(case[0], case[1], case[2], case[3], case[4])

        samples = rebrm_single.get_random_memories(30)
        self.assertEqual(30, len(samples))
        d = self.as_dict(samples)
        r1 = "SRare1" in d
        self.assertTrue(r1)
        r1 = "SRare2" in d
        r2 = "SRare3" in d
        self.assertTrue(r1 or r2)
        r4 = "SRare4" in d
        r5 = "SRare5" in d
        r6 = "SRare6" in d
        self.assertTrue(r4 or r5 or r6)
        return

    #
    # Convert list to dictionary
    #
    @classmethod
    def as_dict(self, lst) -> dict:
        d = dict()
        for e in lst:
            d[e[0]] = e
        return d


#
# Execute the Unit Tests.
#

if __name__ == "__main__":
    if True:
        tests = TestRareEventBiasReplayMemory()
        suite = unittest.TestLoader().loadTestsFromModule(tests)
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(TestRareEventBiasReplayMemory("<unit test function name here>"))
        unittest.TextTestRunner().run(suite)
