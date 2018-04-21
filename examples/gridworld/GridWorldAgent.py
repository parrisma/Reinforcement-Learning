import logging

from reflrn.Agent import Agent
from reflrn.ExplorationStrategy import ExplorationStrategy
from reflrn.State import State


class GridWorldAgent(Agent):

    def __init__(self,
                 agent_id: int,  # immutable & unique id for this agent
                 agent_name: str,  # immutable & unique name for this agent
                 exploration_strategy: ExplorationStrategy,
                 lg: logging):
        self.__lg = lg
        self.__id = agent_id
        self.__name = agent_name
        self.__exploration_strategy = exploration_strategy
        self.__policy = None
        self.__generation = 0

    # Return immutable id
    #
    def id(self):
        return self.__id

    # Return immutable name
    #
    def name(self):
        return self.__name

    #
    # Environment call back when environment shuts down
    #
    def terminate(self):
        pass

    #
    # Environment call back when episode starts
    #
    def episode_init(self, state: State):
        pass

    #
    # Environment call back when episode is completed
    #
    def episode_complete(self, state: State):
        self.__generation += 1
        pass

    #
    # Environment call back to ask the agent to chose an action
    #
    # State : The current state of the environment
    # possible_actions : The set of possible actions the agent can play from this state
    #
    def chose_action(self, state: State, possible_actions: [int]) -> int:
        self.__policy = self.__exploration_strategy.chose_action_policy(self.__name,
                                                                        self.__generation,
                                                                        state,
                                                                        possible_actions)
        return self.__policy.select_action(self.name(), state, possible_actions)

    #
    # Environment call back to reward agent for a play chosen for the given
    # state passed.
    #
    def reward(self, state: State, next_state: State, action: int, reward_for_play: float, episode_complete: bool):
        self.__exploration_strategy.update_strategy(self.name(),
                                                    state,
                                                    next_state,
                                                    action,
                                                    reward_for_play,
                                                    episode_complete)
        return

    #
    # Called by the environment *once* at the start of the session
    # and the action set is given as dictionary
    #
    def session_init(self, actions: dict):
        self.__policy = None
        self.__generation = 0
        return
