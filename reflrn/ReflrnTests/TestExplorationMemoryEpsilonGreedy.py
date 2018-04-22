import logging
import random
import unittest

import numpy as np

from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.ExplorationMemoryEpsilonGreedy import ExplorationMemoryEpsilonGreedy
from reflrn.ReflrnTests.DummyPolicy import DummyPolicy
from reflrn.ReflrnTests.DummyState import DummyState


class TestExplorationMemoryEpsilonGreedy(unittest.TestCase):
    # A Test Object That Cannot Be Found In The Memory (failure case test)
    class NonExistentThing:
        pass

    __lg = None

    __dummy_policy_1 = DummyPolicy()
    __dummy_policy_2 = DummyPolicy()
    __dummy_state_1 = DummyState('DummyState1')
    __dummy_state_2 = DummyState('DummyState2')

    __test_cases = [
        [0, __dummy_policy_1, "AgentOne", __dummy_state_1, __dummy_state_2, 0, float(0), True],

        [1, __dummy_policy_1, "AgentOne", __dummy_state_1, __dummy_state_2, 0, float(0.1), False],
        [1, __dummy_policy_1, "AgentOne", __dummy_state_1, __dummy_state_2, 0, float(0.2), False],
        [1, __dummy_policy_1, "AgentOne", __dummy_state_1, __dummy_state_2, 0, float(0.3), True],
        [2, __dummy_policy_1, "AgentOne", __dummy_state_1, __dummy_state_2, 0, float(0.4), True],
    ]

    @classmethod
    def setUpClass(cls):
        random.seed(42)
        np.random.seed(42)
        cls.__lg = EnvironmentLogging("TestRig-TemporalDifference",
                                      "TestRig-TemporalDifference.log",
                                      logging.DEBUG
                                      ).get_logger()

    def test_single_memory(self):
        test_case_id = 0
        episode_id = int(0)
        emeg = ExplorationMemoryEpsilonGreedy(self.__lg)
        self.__add_test_cases(emeg, self.__test_cases, [test_case_id])

        # Should be only 1 Memory for this episode.
        memory = emeg.get_memories_by_episode(episode=episode_id)
        self.assertEqual(len(memory[0]), 8)
        self.assertEqual(self.__test_case_equal(self.__test_cases[test_case_id], memory[0]), True)

        for mem_type in ExplorationMemoryEpsilonGreedy.SUPPORTED_GETBY_INDEX:
            memory = emeg.get_memories_by_type(mem_type,
                                               self.__test_cases[test_case_id][mem_type])
            self.assertEqual(len(memory), 1)
            self.assertEqual(self.__test_case_equal(self.__test_cases[test_case_id], memory[0]), True)

            memory = emeg.get_memories_by_type(mem_type,
                                               TestExplorationMemoryEpsilonGreedy.NonExistentThing())
            self.assertEqual(None, memory)

        for mem_type in ExplorationMemoryEpsilonGreedy.UNSUPPORTED_GETBY_INDEX:
            self.assertRaises(ExplorationMemoryEpsilonGreedy.MemoryTypeSearchNotSupported,
                              emeg.get_memories_by_type,
                              mem_type,
                              None)

        return

    def test_multi_memory(self):
        start_test_case_id = 1
        episode_id = int(0)
        emeg = ExplorationMemoryEpsilonGreedy(self.__lg)
        self.__add_test_cases(emeg, self.__test_cases, [1, 2, 3, 4])

        # Should be 3 and then 1
        # episode_id
        # (memories expected & start offset in test cases) x 3
        for ep, cnt, st, al, sta, lst, stl in ((1, 3, 1, 4, 1, 1, 4), (2, 1, 4, 4, 1, 1, 4)):
            memory = emeg.get_memories_by_episode(episode=ep)
            self.assertEqual(len(memory), cnt)

            i = st
            for mem in memory:
                self.assertEqual(self.__test_case_equal(self.__test_cases[i], mem), True)
                i += 1

            for mem_type in ExplorationMemoryEpsilonGreedy.SUPPORTED_GETBY_INDEX:
                i = sta
                memory = emeg.get_memories_by_type(mem_type,
                                                   self.__test_cases[i][mem_type],
                                                   last_only=False)
                self.assertEqual(len(memory), al)
                for mem in memory:
                    self.assertEqual(self.__test_case_equal(self.__test_cases[i], mem), True)
                    i += 1

                i = stl
                memory = emeg.get_memories_by_type(mem_type,
                                                   self.__test_cases[i][mem_type],
                                                   last_only=True)
                self.assertEqual(len(memory), lst)
                for mem in memory:
                    self.assertEqual(self.__test_case_equal(self.__test_cases[i], mem), True)
                    i += 1

                memory = emeg.get_memories_by_type(mem_type,
                                                   TestExplorationMemoryEpsilonGreedy.NonExistentThing())
                self.assertEqual(None, memory)

            for mem_type in ExplorationMemoryEpsilonGreedy.UNSUPPORTED_GETBY_INDEX:
                self.assertRaises(ExplorationMemoryEpsilonGreedy.MemoryTypeSearchNotSupported,
                                  emeg.get_memories_by_type,
                                  mem_type,
                                  None)

        return

    #
    # Test equality of expected & actual memory
    #
    @classmethod
    def __test_case_equal(cls,
                          expected: [[], [], [], [], [], [], [], []],
                          actual: [[], [], [], [], [], [], [], []]
                          ) -> bool:
        res = \
            expected[0] == actual[0] and \
            expected[1] == actual[1] and \
            expected[2] == actual[2] and \
            expected[3] == actual[3] and \
            expected[4] == actual[4] and \
            expected[5] == actual[5] and \
            expected[6] == actual[6] and \
            expected[7] == actual[7]
        return res

    #
    # Add the given list of test case entries to the ExplorationMemory
    #
    @classmethod
    def __add_test_cases(cls,
                         emeg: ExplorationMemoryEpsilonGreedy,
                         test_cases: [[object]],
                         test_cases_to_add: [int]) -> None:
        for i in test_cases_to_add:
            emeg.add(episode_id=test_cases[i][ExplorationMemoryEpsilonGreedy.Memory.EPISODE],
                     policy=test_cases[i][ExplorationMemoryEpsilonGreedy.Memory.POLICY],
                     agent_name=test_cases[i][ExplorationMemoryEpsilonGreedy.Memory.AGENT],
                     state=test_cases[i][ExplorationMemoryEpsilonGreedy.Memory.STATE],
                     next_state=test_cases[i][ExplorationMemoryEpsilonGreedy.Memory.NEXT_STATE],
                     action=test_cases[i][ExplorationMemoryEpsilonGreedy.Memory.ACTION],
                     reward=test_cases[i][ExplorationMemoryEpsilonGreedy.Memory.REWARD],
                     episode_complete=test_cases[i][ExplorationMemoryEpsilonGreedy.Memory.EPISODE_COMPLETE])
        return


#
# Execute the ReflrnTests.
#

if __name__ == "__main__":
    if True:
        tests = TestExplorationMemoryEpsilonGreedy()
        suite = unittest.TestLoader().loadTestsFromModule(tests)
        unittest.TextTestRunner().run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(TestExplorationMemoryEpsilonGreedy("test_single_memory"))
        unittest.TextTestRunner().run(suite)
