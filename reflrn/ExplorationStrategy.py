import abc

from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


#
# Exploration Strategy given the iteration number, the state and the possible actions
# returns an action according to the exploration strategy it encodes.
#


class ExplorationStrategy(metaclass=abc.ABCMeta):

    #
    # Given the iteration number, the state and the possible actions
    # returns an action according to the exploration strategy.
    #
    @abc.abstractmethod
    def chose_action(self,
                     agent_name: str,
                     episode_number: int,
                     state: State,
                     possible_actions: [int]) -> int:
        pass

    #
    # return the Policy can be called to select the action. The assumption here is that the
    # implementation is selecting between a number of different policies according to it's
    # defined strategy. In the case the returned policy can be called to get the next action.
    #
    @abc.abstractmethod
    def chose_action_policy(self,
                            agent_name: str,
                            episode_number: int,
                            state: State,
                            possible_actions: [int]) -> Policy:
        pass

    #
    # Update the Exploration strategy based on the reward for the action.
    #
    @abc.abstractmethod
    def update_strategy(self,
                        agent_name: str,
                        state: State,
                        next_state: State,
                        action: int,
                        reward: float,
                        episode_complete: bool):
        pass
