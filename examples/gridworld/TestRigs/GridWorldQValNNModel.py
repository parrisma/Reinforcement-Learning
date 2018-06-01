import logging
from pathlib import Path

import keras
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Activation
from keras.models import Sequential

from examples.gridworld.CannotCloneWeightsOfDifferentModelException import CannotCloneWeightsOfDifferentModelException
from reflrn.Interface.Model import Model


class GridWorldQValNNModel(Model):

    #
    # At init time we need
    # - An arbitrary name for the model, used also for save/load filename.
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
        self.__epochs = 10

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
        # One Input Layer
        # Three Hidden Layers
        # Output layer has no activation as this a a QVal regression net.
        #
        # Used on QVals from a test grid 10 by 10 no regularization is needed and testing
        # with L2.Reg - kernel and dropout decreased the final accuracy even allowing significant training
        # past the point where the cost function had stabilised.
        #
        model.add(Dense(25, input_dim=self.__input_dimension, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(50, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(100, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(200, kernel_initializer='he_uniform'))
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
        self.__bootstrap_model()
        return self.__model.predict_on_batch(x)

    #
    # Given the replay memory train the model
    #
    def train(self, x, y) -> None:
        self.__bootstrap_model()
        self.__model.fit(x=x,
                         y=y,
                         batch_size=self.__batch_size,
                         epochs=self.__epochs,
                         verbose=2,
                         callbacks=[LearningRateScheduler(self.__lr_step_down_decay)])
        self.__inc_lr_epoch()  # count a global fitting call.
        return

    #
    # If the model has not been created or complied, do so.
    #
    def __bootstrap_model(self):
        if self.__model is None:
            self.new_model()
        if not self.__model_compiled:
            self.compile()
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
    # Step Down decay of learning rate - use the global epoch not the local one passed in
    #
    def __lr_step_down_decay(self, _) -> float:
        if self.__lr_epoch % 200 == 0:
            self.__lr -= 0.0001
            self.__lr = max(self.__lr_min, self.__lr)
        return self.__lr

    #
    # Clone the weights of the given model, or throw an exception
    # if it is not an instance of this class as we can only copy
    # weights of identical architecture
    #
    def clone_weights(self, model: 'Model') -> None:
        if not isinstance(model, type(self)):
            raise CannotCloneWeightsOfDifferentModelException(str(type(self)) + " <- " + str(type(model)))
        if self.__model is None:
            raise RuntimeError("Internal Model is value (None) as has not been initialised")
        else:
            self.__model.set_weights(model.get_weights())
        return

    #
    # Return the model weights.
    #
    def get_weights(self):
        if self.__model is None:
            raise RuntimeError("Internal Model is value (None) as has not been initialised")
        return self.__model.get_weights()

    #
    # Save the model using Keras built in save capability.
    #
    def save(self, filename: str) -> None:
        try:
            self.__model.save(filename)
        except Exception as exc:
            err = "Failed to save Keras model to file [" + filename + ": " + str(exc)
            self.__lg.error(err)
            raise RuntimeError(err)
        finally:
            pass
        return

    #
    # Load the current state of the model from a file that was saved
    # from the same Keras model architecture as the new_model method
    # creates.
    #
    def load(self, filename: str) -> None:
        try:
            if Path(filename).is_file():
                model = keras.models.load_model(filename)
            else:
                raise FileExistsError("Keras Model File: [" + filename + "] Not found")
        except Exception as exc:
            err = "Failed to load Keras Deep NN model from file [" + filename + ": " + str(exc)
            self.__lg.error(err)
            raise RuntimeError(err)
        finally:
            pass
        self.__model = model
        return
