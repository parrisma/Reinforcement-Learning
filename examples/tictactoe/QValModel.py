# Class that can predict Q Values based on neural network trained on Q Values
#
import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from reflrn import Persistance


class QValModel:

    def __init__(self):
        self.__model = None
        return

    #
    # Load a saved model. This is fully saved model not just the weights.
    #
    def load(self, keras_saved_model_filename):
        self.__model = keras.models.load_model(keras_saved_model_filename)
        return

    #
    # Create and train a model on teh given Q values and
    # save the full model.
    #
    def train(self, qval_traing_set_filename, model_filename):
        # Load the training data
        p = Persistance()
        X, Y = p.load_as_X_Y(qval_traing_set_filename)

        self.__model = Sequential()
        self.__model.add(Dense(1000, input_dim=10, activation='relu'))
        self.__model.add(Dense(512, activation='relu'))
        self.__model.add(Dense(512, activation='relu'))
        self.__model.add(Dense(512, activation='relu'))
        self.__model.add(Dense(9))

        # Compile model
        self.__model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

        # Fit the model
        self.__model.fit(X, Y, epochs=50, batch_size=10)

        self.__model.save(model_filename)

        scores = self.__model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (self.__model.metrics_names[1], scores[1] * 100))

    #
    # Make an Informed Action given a board state by predicting the QValues
    # from the trained model.
    #
    # The board state must start with the encoded player reference. [1,-1]
    #
    def predicted_q_vals(self, board_state_as_string):
        x = np.zeros((1, 10))
        x[0] = Persistance.x_as_str_to_num_array(board_state_as_string)
        return self.__model.predict(x)
