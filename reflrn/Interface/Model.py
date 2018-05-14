import abc
import numpy as np
from reflrn.Interface.State import State
from reflrn.Interface.ReplayMemory import ReplayMemory


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
    # Run supervised training given the matching input and out put training set.
    #
    #
    @abc.abstractmethod
    def train(self, x, y) -> None:
        pass

    #
    # Predict the Q Values for the action space given the vector encoding of the
    # grid state.
    # ToDo - Throw a custom error if model not present, compiled and trained ?
    #
    @abc.abstractmethod
    def predict(self, state: State) -> [np.float]:
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
