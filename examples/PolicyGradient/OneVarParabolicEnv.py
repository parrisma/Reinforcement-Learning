import copy
import logging
import random

from examples.PolicyGradient.OneVarParabolicState import OneVarParabolicState
from reflrn.Interface.Agent import Agent
from reflrn.Interface.Environment import Environment
from reflrn.Interface.State import State


class OneVarParabolicEnv(Environment):
    min_x = -10
    max_x = +10
    step = 0.1
    action_list = [0, 1]
    actn_dict = {action_list[0]: (+0.1), action_list[1]: (-0.1)}

    def __init__(self,
                 agent: Agent,
                 lg: logging,
                 explain: bool = False,
                 ):
        self.__agent = agent
        self.__lg = lg
        self.__explain = explain
        return

    #
    # We translate integer action into an 'x' value in range -10 to +10 and use x to
    # evaluate -x^2 which is the reward. This is a trivial inverted parabolic curve so
    # to maximise the reward is to find the turning point of the parabola at x = 0.
    #
    @classmethod
    def __reward_function(cls,
                          state: OneVarParabolicState, ) -> float:
        x = state.state_as_array()[0]
        r = -(x * x)
        return r

    def run(self, iterations: int):
        i = 0
        episode_complete = False
        self.__agent.session_init(OneVarParabolicEnv.actn_dict)
        self.__lg.debug("Start ...")
        j = 0
        while i <= iterations:
            state = OneVarParabolicState(round(random.uniform(self.min_x, self.max_x), 1))
            while not episode_complete:
                self.__agent.episode_init(state)
                action = self.__agent.chose_action(state, possible_actions=OneVarParabolicEnv.actions())
                next_state = OneVarParabolicState(self.__play_action(action, state))
                if self.episode_complete(next_state):
                    r = float(0)
                    episode_complete = True
                else:
                    r = self.__reward_function(next_state)
                if self.__explain:
                    self.__lg.debug("S: " + state.state_as_string() +
                                    "  A: " + str(action) +
                                    "  R: " + str(r) +
                                    "  Sn: " + next_state.state_as_string())
                self.__agent.reward(state=state,
                                    next_state=next_state,
                                    action=action,
                                    reward_for_play=r,
                                    episode_complete=False)
                state = next_state
                j += 1
                if j % 500 == 0:
                    self.__lg.debug("Iteration: " + str(i))
                    self.__lg.debug(self.__status())
            j = 0
            i = i + 1
            episode_complete = False
            self.__agent.episode_complete(state)
        self.__lg.debug("Done ...")
        self.__agent.terminate()
        return

    #
    # Current Predicted State
    #
    def __status(self) -> str():
        s = str()
        i = self.min_x
        while i <= self.max_x:
            action = self.__agent.chose_action(OneVarParabolicState(i),
                                               possible_actions=OneVarParabolicEnv.actions())
            if i == 0:
                s = s + '|'
            if action == 0:
                s = s + '>'
            else:
                s = s + '<'
            i += 0.5
        return s

    #
    # Make the play chosen by the agent.
    #
    def __play_action(self,
                      action: int,
                      state: OneVarParabolicState) -> float:
        # Make the play on the board.
        new_x = state.state_as_array()[0]
        new_x += self.actn_dict[action]
        return new_x

    @classmethod
    def actions(cls, _: State = None) -> [int]:
        return copy.deepcopy(OneVarParabolicEnv.action_list)

    def episode_complete(self,
                         state: State = None) -> bool:
        if state is not None:
            x = state.state_as_array()[0]
        else:
            raise ValueError("State must be supplied and not passed as None")
        return x < self.min_x or x > self.max_x

    def save(self, file_name: str) -> None:
        raise NotImplementedError()

    def load(self, file_name: str):
        raise NotImplementedError()

    def import_state(self, state: str):
        raise NotImplementedError()

    def export_state(self) -> str:
        raise NotImplementedError()

    def attributes(self) -> dict:
        raise NotImplementedError()

    def state(self) -> State:
        raise NotImplementedError()

    def attribute_names(self) -> [str]:
        raise NotImplementedError()

    def attribute(self, attribute_name: str) -> object:
        raise NotImplementedError()
