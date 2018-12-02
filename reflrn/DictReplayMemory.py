import logging
import random

import numpy as np

from reflrn.Interface.ReplayMemory import ReplayMemory
from reflrn.Interface.State import State


#
# Manage the shared replay memory between {n} actors in an Actor/Critic model.
#
# ToDo: Consider https://github.com/robtandy/randomdict as a non functional improvement
#

class DictReplayMemory(ReplayMemory):
    # Memory List Entry Off Sets
    mem_episode_id = 0
    mem_state = 1
    mem_next_state = 2
    mem_action = 3
    mem_reward = 4
    mem_complete = 5

    def __init__(self,
                 lg: logging,
                 replay_mem_size: int
                 ):
        self.__replay_memory = dict()
        self.__replay_mem_size = replay_mem_size
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

        # Add the memory, if the same memory (by state) exists then remove it before adding the more
        # recent memory.
        #
        sas = state.state_as_string()
        if sas in self.__replay_memory:
            del self.__replay_memory[sas]

        if len(self.__replay_memory) >= self.__replay_mem_size:
            # remove random element
            rndk = random.choice(list(self.__replay_memory.keys()))
            del self.__replay_memory[rndk]

        self.__replay_memory[sas] = (self.__episode_id, state, next_state, action, reward, episode_complete)

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
        for k in np.random.choice(list(self.__replay_memory.keys()), min(ln, sample_size)):
            samples.append(self.__replay_memory[k])

        # Ensure results are random order
        return random.sample(samples, len(samples))

    def get_last_memory(self, state: State = None) -> [int, State, State, int, float, bool]:
        raise RuntimeError("get_last_memory, method not implemented")
