import abc
import numpy as np


#
# This abstract base class that is the contract for creating Keras NN Models.
#


class Model(metaclass=abc.ABCMeta):

    #
    # Return a Keras model, There is no expectation about the NN architecture.
    #
    @abc.abstractmethod
    def new_model(self):
        pass

    #
    # Compile the given Keras model.
    #
    @abc.abstractmethod
    def compile(self):
        pass

    #
    # Predict the Q Values for the action space given the vector encoding of the
    # grid state.
    #
    @abc.abstractmethod
    def predict(self, state_encoding: [np.float]) -> [np.float]:
        pass

    #
    # Save the current state of the model to a file.
    #
    @abc.abstractmethod
    def save(self, filename: str):
        pass

    #
    # Load the current state of the model from a file that was saved
    # from the same Keras model architecture as the new_model method
    # creates.
    #
    @abc.abstractmethod
    def load(self, filename: str):
        pass
