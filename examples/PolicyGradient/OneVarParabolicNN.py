import copy

from keras.layers import Dense, Activation
from keras.models import Sequential

from reflrn.Interface.NeuralNetwork import NeuralNetwork


class OneVarParabolicNN(NeuralNetwork):

    def __init__(self,
                 input_dimension: int,
                 output_dimension: int):
        self.__input_dimension = input_dimension
        self.__output_dimension = output_dimension
        return

    #
    # A sufficient complexity to learn the mapping of a single var x -> f(x^2)
    #
    def new(self):
        model = Sequential()
        # One Input Layer
        # Three Hidden Layers
        # Output layer has no activation as this is a QVal regression net.
        #
        model.add(Dense(10, input_dim=self.__input_dimension, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(50, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(10, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(units=self.__output_dimension, kernel_initializer='normal'))
        return model

    def input_dimension(self) -> int:
        return copy.copy(self.__input_dimension)

    def output_dimension(self) -> int:
        return copy.copy(self.__output_dimension)
