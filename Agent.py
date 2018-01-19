import abc
from State import State

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
    # given the environment in the current state.
    #
    @abc.abstractmethod
    def chose_action(self, state: State)-> int:
        pass

    #
    # Environment call back to reward agent for a playing the chosen
    # action from the chose_action() method. The environment is given
    # in the state **after** the action was played.
    #
    @abc.abstractmethod
    def reward(self, state: State, reward_for_play: float):
        pass
