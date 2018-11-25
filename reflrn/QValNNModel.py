import logging
from pathlib import Path

import keras
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Activation
from keras.models import Sequential

from reflrn.Interface.Model import Model
from reflrn.Interface.ModelParams import ModelParams
from reflrn.exceptions.CannotCloneWeightsOfDifferentModelException import CannotCloneWeightsOfDifferentModelException


class QValNNModel(Model):

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
                 lg: logging,
                 model_params: ModelParams):
        self.__lg = lg
        self.__agent_name = model_name
        self.__input_dimension = input_dimension
        self.__num_actions = num_actions
        self.__model = None
        self.__batch_size = model_params.get_parameter(ModelParams.batch_size)
        self.__model_compiled = False
        self.__epochs = 10

        self.__lr_0 = model_params.get_parameter(ModelParams.learning_rate_0)
        self.__lr_min = model_params.get_parameter(ModelParams.learning_rate_min)
        self.__lr = model_params.get_parameter(ModelParams.learning_rate_0)
        self.__lr_epoch = 1
        self.__verbose = model_params.get_parameter(ModelParams.verbose)

        return

    #
    # Return an un-compiled Keras model that has an architecture capable of learning
    # q-values of number and complexity for a "simple" problem given the number of
    # actions and specified input dimension.
    #
    def new_model(self):
        model = Sequential()
        # One Input Layer
        # Three Hidden Layers
        # Output layer has no activation as this a a QVal regression net.
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
        self.__model.compile(loss='mean_squared_error',
                             optimizer='adam',
                             metrics=['accuracy'])
        self.__model_compiled = True
        return

    #
    # Predict the Q Values for the action space given the vector encoding of the
    # grid curr_coords.
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
                         verbose=self.__verbose,
                         callbacks=[LearningRateScheduler(self.__lr_step_down_decay)])
        self.__inc_lr_epoch()  # count a global fitting call.
        return

    #
    # Evaluate the performance of the model
    #
    def evaluate(self, x, y):
        scores = self.__model.evaluate(x,
                                       y,
                                       verbose=0)
        return scores

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
    # Reset the learning curr_coords of the model.
    #
    def reset(self) -> None:
        self.__lr = self.__lr_0
        self.__lr_epoch = 1

    #
    # Increment the global episode count. This can be used when the learning rate etc are driven
    # by an episode cycle that is not linked directly to model training calls.
    #
    def __inc_lr_epoch(self):
        self.__lr_epoch += 1

    #
    # Step Down decay of learning rate - use the global episode not the local one passed in
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
    # Load the current curr_coords of the model from a file that was saved
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
