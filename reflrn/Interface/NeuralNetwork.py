import abc


#
# This abstract base class that builds a (Keras) neural network.
#


class NeuralNetwork(metaclass=abc.ABCMeta):
    #
    # return a build but not compiled Keras model.
    #
    @abc.abstractmethod
    def new(self):
        pass

    #
    # Input Dimension
    #
    @abc.abstractmethod
    def input_dimension(self) -> int:
        pass

    #
    # Output Dimension
    #
    @abc.abstractmethod
    def output_dimension(self) -> int:
        pass
