import abc

from reflrn.Interface.State import State


#
# This abstract base class that is the contract for updating and interpreting the policy.
#


class Policy(metaclass=abc.ABCMeta):

    #
    # Update Policy with the give state, action, reward details.
    #
    # prev_state : the previous state for this Agent; None if no previous state
    # prev_action : the previous action of this agent; has no meaning is prev_state = None
    # state : current state of the environment *after* the given action was played
    # action : the action played by this agent that moved the state to the state passed
    # reward : the reward associated with the given state/action pair.
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
    # return the action that has the highest (current) pay-off given the current state.
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
