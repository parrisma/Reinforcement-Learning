import abc
from Environment import Environment

#
# This abstract class provides an immutable state representation of the environment
# it was constructed with.
#


class State(metaclass=abc.ABCMeta):

    #
    # An environment specific representation for Env. State
    #
    @abc.abstractmethod
    def state(self) -> object:
        pass


    #
    # An string representation of the environment state
    #
    @abc.abstractmethod
    def state_as_string(self) -> str:
        pass

    #
    # The environment the state was initiated with.
    #
    @abc.abstractmethod
    def get_env(self) -> Environment:
        pass
