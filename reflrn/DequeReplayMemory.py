import logging
import random
from collections import deque

import numpy as np

from reflrn.Interface.ReplayMemory import ReplayMemory
from reflrn.Interface.State import State


#
# Manage the shared replay memory between {n} actors in an Actor/Critic model.
#

class DequeReplayMemory(ReplayMemory):
    # Memory List Entry Off Sets
    mem_episode_id = 0
    mem_state = 1
    mem_next_state = 2
    mem_action = 3
    mem_reward = 4
    mem_complete = 5

    def __init__(self, lg: logging, replay_mem_size: int):
        self.__replay_memory = deque([], maxlen=replay_mem_size)
        self.__episode_id = 0
        self.__lg = lg
        return

    #
    # Add a memory to the reply memory, but tag it with the episode id such that whole episodes
    # can later be recovered for training.
    #
    def append_memory(self,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool) -> None:

        # Track the SAR (State Action Reward) for critic training.
        # Must match order as defined by class level mem_<?> offsets.
        self.__replay_memory.append((self.__episode_id, state, next_state, action, reward, episode_complete))
        if episode_complete:
            self.__episode_id += 1
        return

    #
    # How many items in the replay memory deque
    #
    def len(self) -> int:
        return len(self.__replay_memory)

    #
    # Get a random set of memories buy taking sample_size random samples and then
    # returning the whole episode for each random sample.
    #
    # return list of elements [episode, curr_state, next_state, action, reward, complete]
    #
    # ToDo: Whole Episodes = False
    #
    def get_random_memories(self,
                            sample_size: int,
                            whole_episodes: bool = False) -> [[int, State, State, int, float, bool]]:
        ln = self.len()
        samples = list()
        indices = np.random.choice(ln, min(ln, sample_size), replace=False)
        for idx in indices:
            memory = self.__replay_memory[idx]
            if whole_episodes:
                # get_memories_by_type the whole episode. Look forward and back until episode id changes
                episode_id = memory[DequeReplayMemory.mem_episode_id]
                episode_deque = deque([])
                episode_deque.append(memory)

                i = idx - 1
                while i >= 0 and self.__replay_memory[i][DequeReplayMemory.mem_episode_id] == episode_id:
                    episode_deque.appendleft(self.__replay_memory[i])
                    i -= 1

                i = idx + 1
                while i < ln and self.__replay_memory[i][DequeReplayMemory.mem_episode_id] == episode_id:
                    episode_deque.append(self.__replay_memory[i])
                    i += 1

                while len(samples) < sample_size:
                    samples.append(list(episode_deque.pop()))
            else:
                samples.append(memory)

        # Ensure results are random order
        return random.sample(samples, len(samples))

    def get_last_memory(self, state: State = None) -> [int, State, State, int, float, bool]:
        raise RuntimeError("get_last_memory, method not implemented")
