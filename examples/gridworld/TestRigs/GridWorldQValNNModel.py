import logging
import numpy as np

from reflrn.Interface.Model import Model


class GridWorldQValNNModel(Model):

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self,
                 model_name: str,
                 input_dimension: int,
                 num_actions: int,
                 lg: logging):
        self.__lg = lg
        self.__agent_name = model_name
        return

    #
    # Return an un-compiled Keras model that has an architecture capable of learning
    # q-values of number and complexity for a grid-world problem given the number of
    # actions and specified input dimension.
    #
    def new_model(self):
        pass

    #
    # Compile the model with an appropriate loss function and optimizer.
    #
    def compile(self):
        pass

    #
    # Predict the Q Values for the action space given the vector encoding of the
    # grid state.
    #
    def predict(self, state_encoding: [np.float]) -> [np.float]:
        pass

    #
    # Save the model using Keras built in save capability.
    #
    def save(self, filename: str):
        pass

    #
    # Load the current state of the model from a file that was saved
    # from the same Keras model architecture as the new_model method
    # creates.
    #
    def load(self, filename: str):
        pass
