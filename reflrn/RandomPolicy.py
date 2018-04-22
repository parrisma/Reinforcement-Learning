import sys
from random import randint
from random import shuffle

from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


class RandomPolicy(Policy):

    def __init__(self,
                 prefer_new: bool = False  # Try to explore new state/action before re-playing
                 ):
        self.__prefer_new = prefer_new
        self.__trace = dict()

    #
    # This is a pure random policy, just pick any of the possible actions. If prefer_new is true an action is
    # chosen that has been least visited from the current state.
    #
    def select_action(self,
                      agent_name: str,
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
                if self.__key(agent_name, state, a) in self.__trace:
                    ct = self.__trace[self.__key(agent_name, state, a)]
                if actn is None:
                    actn = a
                    mn = ct
                else:
                    if ct < mn:
                        actn = a
                        mn = ct
            if actn is None:
                actn = possible_actions[randint(0, len(possible_actions) - 1)]
            if self.__key(agent_name, state, actn) not in self.__trace:
                self.__trace[self.__key(agent_name, state, actn)] = 0
            self.__trace[self.__key(agent_name, state, actn)] += 1

        else:
            actn = possible_actions[randint(0, len(possible_actions) - 1)]

        return actn

    def __key(self, agent_name: str, state: State, action: int):
        return agent_name + ':' + state.state_as_string() + ':' + str(action)

    def update_policy(self, agent_name: str, state: State, next_state: State, action: int, reward: float,
                      episode_complete: bool):
        pass

    def save(self, filename: str = None):
        pass

    def load(self, filename: str = None):
        pass
