import logging

import numpy as np
from keras import regularizers
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler

from reflrn.Interface.Model import Model


class GridWorldQValNNModel(Model):

    #
    # At init time we need
    # - An arbitrary name for the model (debug only)
    # - The NN input dimension.
    # - The max number of possible actions from a grid location.
    # - A logging channel.
    #
    def __init__(self,
                 model_name: str,
                 input_dimension: int,
                 num_actions: int,
                 num_grid_cells: int,
                 lg: logging,
                 batch_size: int = 32,
                 num_epoch: int = 1,
                 lr_0: float = 0.001,
                 lr_min: float = 0.001):
        self.__lg = lg
        self.__agent_name = model_name
        self.__input_dimension = input_dimension
        self.__num_actions = num_actions
        self.__num_grid_cells = num_grid_cells
        self.__model = None
        self.__batch_size = batch_size
        self.__num_epoch = num_epoch
        self.__model_compiled = False

        self.__lr_0 = lr_0
        self.__lr_min = lr_min
        self.__lr = lr_0
        self.__lr_epoch = 1

        return

    #
    # Return an un-compiled Keras model that has an architecture capable of learning
    # q-values of number and complexity for a grid-world problem given the number of
    # actions and specified input dimension.
    #
    def new_model(self):
        model = Sequential()
        # Base layer sizes off num-actions and num-grid-cells
        # for small grids this is reasonable, for more complex grids
        # a custom model will most probably be required.
        l0_size = self.__num_actions * self.__num_grid_cells
        ln_size = int(l0_size / 2)
        model.add(Dense(250, input_dim=self.__input_dimension, kernel_initializer='normal'))
        model.add(Activation('relu'))
        model.add(Dense(500, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dense(units=self.__num_actions, kernel_initializer='normal'))
        self.__model = model
        return self.__model

    #
    # Compile the model with an appropriate loss function and optimizer.
    #
    def compile(self):
        self.__model.compile(loss='mean_squared_error', optimizer='adam')
        self.__model_compiled = True
        return

    #
    # Predict the Q Values for the action space given the vector encoding of the
    # grid state.
    #
    def predict(self, x) -> [np.float]:
        if self.__model is not None and self.__model_compiled:
            return self.__model.predict_on_batch(x)

    #
    # Given the replay memory train the model
    #
    def train(self, x, y) -> None:
        self.__model.fit(x=x, y=y, batch_size=16, epochs=1, verbose=0,
                         callbacks=[LearningRateScheduler(self.__lr_step_down_decay)])
        self.__inc_lr_epoch()  # count a global fitting call.
        return

    #
    # Reset the learning state of the model.
    #
    def reset(self) -> None:
        self.__lr = self.__lr_0
        self.__lr_epoch = 1

    #
    # Increment the global epoch count. This can be used when the learning rate etc are driven
    # by an epoch cycle that is not linked directly to model training calls.
    #
    def __inc_lr_epoch(self):
        self.__lr_epoch += 1

    #
    # Step Down decay of learning rate - use the global epoch not the local one passed int
    #
    def __lr_step_down_decay(self, _) -> float:
        if self.__lr_epoch % 200 == 0:
            self.__lr -= 0.0001
            self.__lr = max(self.__lr_min, self.__lr)
        return self.__lr

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
