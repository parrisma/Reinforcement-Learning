import abc

from reflrn.Interface.State import State


#
# This is an interface specification for an reinforcement learning agent
#


class Agent(metaclass=abc.ABCMeta):

    #
    # Immutable Id for the Agent
    #
    @abc.abstractmethod
    def id(self) -> int:
        pass

    #
    # Immutable Name for the Agent
    #
    @abc.abstractmethod
    def name(self) -> str:
        pass

    #
    # Called by the environment *once* at the start of the session
    # and the action set is given as dictionary
    #
    @abc.abstractmethod
    def session_init(self, actions: dict):
        pass

    #
    # Environment call back when environment shuts down
    #
    @abc.abstractmethod
    def terminate(self):
        pass

    #
    # Environment call back when episode starts
    #
    @abc.abstractmethod
    def episode_init(self, state: State):
        pass

    #
    # Environment call back when episode is completed
    #
    @abc.abstractmethod
    def episode_complete(self, state: State):
        pass

    #
    # Environment call back to ask the agent to chose an action
    #
    # State : The current state of the environment
    # possible_actions : The set of possible actions the agent can play from this state
    #
    @abc.abstractmethod
    def chose_action(self, state: State, possible_actions: [int]) -> int:
        pass

    #
    # The callback via which the environment informs the agent of a reward as a result of an action.
    # state     : The state *before* the action is taken : S
    # next_state: The State after the action is taken : S'
    # action    : The action that transitioned S to S'
    # reward    : The reward for playing action in state S
    # episode_complete : If environment is episodic, then true if the reward relates to the last reward in an episode.
    #
    @abc.abstractmethod
    def reward(self,
               state: State,
               next_state: State,
               action: int,
               reward_for_play: float,
               episode_complete: bool):
        pass
