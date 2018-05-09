import abc


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
    # State encoded as a numpy array that can be passed as the X (input) into
    # a Neural Net. The dimensionality can vary depending on the implementation
    # from a linear vector for a simple Sequential model to an 3D array for a
    # multi layer convolutional model.
    #
    @abc.abstractmethod
    def state_as_array(self):
        pass
