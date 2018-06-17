import logging
import random
from collections import deque
from random import shuffle

import numpy as np

from reflrn.Interface.ReplayMemory import ReplayMemory
from reflrn.Interface.State import State


#
# Establish which rewards are rare and bias the returned sample memories to include
# those rewards to allow the network to learn about these rare events.
#
# ToDo: For non-stationary problem space add capability to overwrite / delete rare bias_memories
# that lose their rare status
#

class RareEventBiasReplayMemory(ReplayMemory):
    mem_state = 0
    mem_next_state = 1
    mem_action = 2
    mem_reward = 3
    mem_complete = 4

    def __init__(self,
                 lg: logging,
                 replay_mem_size: int = 2000):
        self.lg = lg
        self.replay_mem_size = replay_mem_size
        self.core_memory = deque([], maxlen=self.replay_mem_size)
        self.stddev = np.float(0)
        self.bias_memory = dict()
        self.bias_std = np.float(0)
        self.bias_avg = np.float(0)
        self.bias_prob = None

        return

    #
    # Add the given memory to the correct std factor bucket.
    #
    def _add_bias_mem(self, mem):
        cur_state, next_state, action, reward, done = mem
        std_factor = self._bias_factor(reward)
        if std_factor not in self.bias_memory:
            self.bias_memory[std_factor] = deque([], maxlen=self.replay_mem_size)
        self.bias_memory[std_factor].append(mem)
        return

    def _bias_factor(self, reward: np.float) -> int:
        return np.absolute(np.round((reward - self.bias_avg) / self.bias_std))

    #
    # Has there been a material shift in the std deviation of the rewards
    #
    def _std_shifted(self, new_std) -> bool:
        if self.bias_std == 0:
            return True
        return np.absolute((self.bias_std - new_std) / self.bias_std) >= 0.1

    #
    # Update biasing stats.
    #
    def _update_bias_memory(self, new_mem):
        l = len(self.core_memory)
        cm = np.array(self.core_memory)[:, self.mem_reward]  # extract the reward column
        cm = list(map(lambda x: np.absolute(np.float(x)), cm))
        std = np.std(cm)
        if std == 0:
            std = np.float(1)
        avg = np.average(cm)
        if self._std_shifted(std):
            self.bias_std = std
            self.bias_avg = avg
            self.bias_memory = dict()
            for mem in self.core_memory:
                self._add_bias_mem(mem)
            self.bias_prob = dict()
        else:
            self._add_bias_mem(new_mem)

        for k in self.bias_memory.keys():
            self.bias_prob[k] = len(self.bias_memory[k]) / len(self.core_memory)

        return

    #
    # Add the given memory to the core memory and update the "rare" memory stats as needed
    #
    def append_memory(self,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool) -> None:
        self.core_memory.append((state, next_state, action, reward, episode_complete))
        self._update_bias_memory((state, next_state, action, reward, episode_complete))
        return

    def len(self) -> int:
        return len(self.core_memory)

    #
    # Select a set of random memories equal in number to teh given sample_size
    #
    def get_random_memories(self,
                            sample_size: int) -> [[], [], [], [], [], []]:
        if len(self.core_memory) < sample_size:
            raise RareEventBiasReplayMemory.SampleMemoryTooSmall("Current memory is empty or smaller than sample size")

        if sample_size < len(self.bias_prob):
            raise RareEventBiasReplayMemory.SampleSizeSmallerThanNumberOfBiases("Sample Size too small")

        samples = None
        d = self.probabilities_to_sub_sample_sizes(sample_size)
        for k in d.keys():
            samples_p = random.sample(self.bias_memory[k], d[k])
            if samples is None:
                samples = samples_p
            else:
                for s in samples_p:
                    if len(samples) < sample_size:
                        samples.append(s)
        shuffle(samples)
        return samples

    #
    # Convert the sample probabilities to proportions of the sample size, such that
    # the total is equal to the sample size.
    #
    def probabilities_to_sub_sample_sizes(self, sample_size: int) -> dict:
        keys = sorted(self.bias_prob.keys(), reverse=True)
        ss = np.zeros(len(keys))
        i = 0
        for k in keys:
            ss[i] = int(self.bias_prob[k] * sample_size)
            i += 1

        tot = 0
        for i in range(0, len(ss)):
            if 0 == ss[i]:
                ss[i] = 1
            tot += ss[i]

        if tot != sample_size:
            i = np.argmax(ss)
            ss[i] += (sample_size - tot)
            if ss[i] < 0:
                raise RareEventBiasReplayMemory.SampleSizeSmallerThanNumberOfBiases("Sample Size too small.")

        d = dict()
        i = 0
        for k in keys:
            if ss[i] > 0:
                d[k] = int(ss[i])
            i += 1

        return d

    #
    # How many entries in reply memory
    #
    def get_num_memories(self):
        return len(self.core_memory)

    #
    # Get just the last memory with respect to the given state. If given state is
    # None return the last memory overall.
    #
    def get_last_memory(self, state: State = None) -> [[], [], [], [], [], []]:
        lst = None
        if self.core_memory is not None and len(self.core_memory) > 0:
            if state is None:
                lst = self.core_memory[len(self.core_memory) - 1]
            else:
                for i in range(len(self.core_memory) - 1, 0, -1):
                    if self.core_memory[i][0] == state:
                        if i > 0:
                            lst = self.core_memory[i - 1]
                        break
        return lst

    class SampleSizeSmallerThanNumberOfBiases(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    class SampleMemoryTooSmall(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)
