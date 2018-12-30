import logging

from examples.gridworld.RenderGridWorldAgentExplorationMemory import RenderGridWorldAgentExplorationMemory
from reflrn.AgentExplorationMemory import AgentExplorationMemory
from reflrn.Interface.Agent import Agent
from reflrn.Interface.ExplorationStrategy import ExplorationStrategy
from reflrn.Interface.State import State


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
        self.__episode = 0
        self.__exploration_memory = AgentExplorationMemory(self.__lg)

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
        self.__lg.info(RenderGridWorldAgentExplorationMemory(self.__lg, True).render_episode(self.__exploration_memory,
                                                                                             self.__episode))
        self.__episode += 1
        pass

    #
    # Environment call back to ask the agent to chose an action
    #
    # State : The current curr_coords of the environment
    # possible_actions : The set of possible actions the agent can play from this curr_coords
    #
    def chose_action(self, state: State, possible_actions: [int]) -> int:
        self.__policy = self.__exploration_strategy.chose_action_policy(self.__name,
                                                                        self.__episode,
                                                                        state,
                                                                        possible_actions)
        return self.__policy.select_action(self.name(), state, possible_actions)

    #
    # Environment call back to reward agent for a play chosen for the given
    # curr_coords passed.
    #
    def reward(self, state: State, next_state: State, action: int, reward_for_play: float, episode_complete: bool):
        self.__exploration_strategy.update_strategy(self.name(),
                                                    state,
                                                    next_state,
                                                    action,
                                                    reward_for_play,
                                                    episode_complete)

        self.__update_exploration_memory(self.name(),
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
        self.__episode = 0
        return

    #
    # Update the exploration memory so we can track stats.
    #
    def __update_exploration_memory(self,
                                    agent_name: str,
                                    state: State,
                                    next_state: State,
                                    action: int,
                                    reward: float,
                                    episode_complete: bool) -> None:
        self.__exploration_memory.add(episode_id=self.__episode,
                                      policy=self.__policy,
                                      agent_name=agent_name,
                                      state=state,
                                      next_state=next_state,
                                      action=action,
                                      reward=reward,
                                      episode_complete=episode_complete)
        return

    @property
    def explain(self) -> bool:
        False
