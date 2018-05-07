import itertools
import logging
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from reflrn.Interface.ExplorationMemory import ExplorationMemory
from reflrn.Interface.RenderExplorationMemories import RenderExplorationMemory


class RenderGridWorldAgentExplorationMemory(RenderExplorationMemory):
    __plot_pause = 0.0001

    def __init__(self,
                 lg: logging,
                 do_plot: bool = False,
                 replay_mem_size: int = 10000):
        self.__episode_memory = deque([], maxlen=replay_mem_size)
        self.__do_plot = do_plot
        self.__fig = None
        self.__lg = lg
        return

    def render_episode(self, exploration_memory: ExplorationMemory, episode: int) -> str:

        memory = exploration_memory.get_memories_by_episode(episode=episode)
        ep_cost = float(0)
        s = ''
        if memory is not None:
            ep_len = len(memory)
            for mem in memory:
                ep_cost += mem[ExplorationMemory.Memory.REWARD]
            s = 'Episode Summary : Length: ' + str(ep_len) + ' Cost : ' + str(ep_cost)
            self.__episode_memory.append([s, ep_cost])

        if self.__do_plot:
            #self.__plot()
            pass

        return s

    def __plot(self):
        if self.__fig is None:
            self.__fig = plt.figure()

        ln = min(len(self.__episode_memory), 100)
        x = np.arange(0, ln, 1)
        y = list(itertools.islice(self.__episode_memory, max(0, ln - 100), ln))
        y1 = []
        y2 = []
        for v in y:
            y1.append(v[0])
            y2.append(v[1])
        ax = self.__fig.gca()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.plot(x, y1)
        ax.plot(x, y2)
        plt.pause(self.__plot_pause)
        plt.show(block=False)
        plt.gcf().clear()
        return
