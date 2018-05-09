import logging
import random

import numpy as np

from reflrn.Interface.ExplorationStrategy import ExplorationStrategy
from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


class EpsilonGreedyExplorationStrategy(ExplorationStrategy):

    #
    # This is teh simple epsilon greedy strategy the flips between a greedy action
    # and a random action when a random variable is above a given threshold (epsilon)
    #
    def __init__(self,
                 greedy_policy: Policy,
                 exploration_policy: Policy,
                 epsilon: np.float,
                 lg: logging):
        self.__greedy_policy = greedy_policy
        self.__exploration_policy = exploration_policy
        self.__epsilon = epsilon
        self.__lg = lg

    #
    # Chose between the greedy policy and the exploration policy based on
    # the epsilon threshold and calls that policy to get_memories_by_type the next action.
    # greedy policy is selected unless the random variable is > epsilon
    #
    def chose_action(self,
                     agent_name: str,
                     episode_number: int,
                     state: State,
                     possible_actions: [int]) -> int:
        action = self.chose_action_policy(agent_name,
                                          episode_number,
                                          state,
                                          possible_actions).select_action(agent_name,
                                                                          state,
                                                                          possible_actions)

        self.__lg.debug(agent_name + "chosen action: " + str(action))
        return action

    #
    # Chose between the greedy policy and the exploration policy based on
    # the epsilon threshold and returns the Policy for the caller to call.
    #
    def chose_action_policy(self,
                            agent_name: str,
                            episode_number: int,
                            state: State,
                            possible_actions: [int]) -> Policy:

        if random.random() < self.__epsilon_decay(episode_number):
            # if random.random() > self.__epsilon:
            self.__lg.debug(agent_name + " exploration policy selected")
            return self.__exploration_policy
        else:
            self.__lg.debug(agent_name + " greedy policy selected")
            return self.__greedy_policy

    #
    # All updates are feed to the greedy policy as the greedy policy is managing the q values.
    #
    def update_strategy(self,
                        agent_name: str,
                        state: State,
                        next_state: State,
                        action: int,
                        reward: float,
                        episode_complete: bool):

        self.__greedy_policy.update_policy(agent_name,
                                           state,
                                           next_state,
                                           action,
                                           reward,
                                           episode_complete)

    def __epsilon_decay(self,
                        episode_num: int) -> float:
        initial = 1.0
        k = .01
        new_ep = initial * np.exp(-k * episode_num)
        self.__lg.debug("New Epsilon :" + str(new_ep))
        if new_ep < 0.05:
            new_ep = 0.05
        return new_ep
