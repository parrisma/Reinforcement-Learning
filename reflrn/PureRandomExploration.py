import sys
from random import randint
from random import shuffle

from reflrn.Interface.Agent import Agent
from reflrn.Interface.ExplorationPlay import ExplorationPlay
from reflrn.Interface.State import State


class PureRandomExploration(ExplorationPlay):

    def __init__(self,
                 prefer_new: bool = False  # Try to explore new curr_coords/action before re-playing
                 ):
        self.__prefer_new = prefer_new
        self.__trace = dict()

    #
    # This is a pure random play, just pick any of the possible actions.
    #
    def select_action(self,
                      agent: Agent,
                      state: State,
                      possible_actions: [int]
                      ) -> int:
        actn = None
        mn = sys.maxsize
        if self.__prefer_new:
            pa = possible_actions[:]
            shuffle(pa)
            for a in pa:
                ct = 0
                if self.__key(agent, state, a) in self.__trace:
                    ct = self.__trace[self.__key(agent, state, a)]
                if actn is None:
                    actn = a
                    mn = ct
                else:
                    if ct < mn:
                        actn = a
                        mn = ct
            if actn is None:
                actn = possible_actions[randint(0, len(possible_actions) - 1)]
            if self.__key(agent, state, actn) not in self.__trace:
                self.__trace[self.__key(agent, state, actn)] = 0
            self.__trace[self.__key(agent, state, actn)] += 1

        else:
            actn = possible_actions[randint(0, len(possible_actions) - 1)]

        return actn

    def __key(self, agent: Agent, state: State, action: int):
        return str(agent.id()) + ':' + state.state_as_string() + ':' + str(action)
