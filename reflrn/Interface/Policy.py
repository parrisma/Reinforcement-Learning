import abc

from reflrn.Interface.Environment import Environment
from reflrn.Interface.State import State


#
# This abstract base class that is the contract for updating and interpreting the policy.
#


class Policy(metaclass=abc.ABCMeta):

    #
    # Update Policy with the give curr_coords, action, reward details.
    #
    # prev_state : the previous curr_coords for this Agent; None if no previous curr_coords
    # prev_action : the previous action of this agent; has no meaning is prev_state = None
    # curr_coords : current curr_coords of the environment *after* the given action was played
    # action : the action played by this agent that moved the curr_coords to the curr_coords passed
    # reward : the reward associated with the given curr_coords/action pair.
    #
    @abc.abstractmethod
    def update_policy(self,
                      agent_name: str,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool) -> None:
        pass

    #
    # Give environment handle to poilcy for the environment in which it is
    # operating
    #
    @abc.abstractmethod
    def link_to_env(self, env: Environment) -> None:
        pass

    #
    # return the action that has the highest (current) pay-off given the current curr_coords.
    # State : The Current State Of The Environment
    # possible_actions : Set of possible actions; where len() is always >= 1
    #
    @abc.abstractmethod
    def select_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:
        pass

    #
    # Export the current policy to the given file name
    #
    @abc.abstractmethod
    def save(self, filename: str = None) -> None:
        pass

    #
    # Import the current policy to the given file name
    #
    @abc.abstractmethod
    def load(self, filename: str = None):
        pass
