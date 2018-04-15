import logging
from collections import deque

import numpy as np

from reflrn.State import State


#
# Manage the shared replay memory between {n} actors in an Actor/Critic model.
#

class ReplayMemory:
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
    def appendMemory(self,
                     state: State,
                     next_state: State,
                     action: int,
                     reward: float,
                     episode_complete: bool):

        # Track the SAR (State Action Reward) for critic training.
        # Must match order as defined by class level mem_<?> offsets.
        self.__replay_memory.append((self.__episode_id, state, next_state, action, reward, episode_complete))
        if episode_complete:
            self.__episode_id += 1
        return

    #
    # How many items in the replay memory deque
    #
    def len(self):
        return len(self.__replay_memory)

    #
    # Get a random set of memories bu taking sample_size random samples and then
    # returning the whole episode for each random sample.
    #
    def getRandomMemories(self, sample_size: int):
        ln = self.len()
        indices = np.random.choice(ln, min(ln, sample_size), replace=False)
        cols = [[], [], [], [], [], []]  # episode, state, next_state, action, reward, complete
        for idx in indices:
            memory = self.__replay_memory[idx]
            # get the whole episode. Look forward and back until episode id changes
            episode_id = memory[ReplayMemory.mem_episode_id]
            episode_deque = deque([])
            episode_deque.append(memory)
            i = idx - 1
            while i >= 0 and self.__replay_memory[i][ReplayMemory.mem_episode_id] == episode_id:
                episode_deque.appendleft(self.__replay_memory[i])
                i -= 1

            i = idx + 1
            while i < ln and self.__replay_memory[i][ReplayMemory.mem_episode_id] == episode_id:
                episode_deque.append(self.__replay_memory[i])
                i += 1

            # Add Whole Episode
            for mem in episode_deque:
                for col, value in zip(cols, mem):
                    col.append(value)
        return cols
