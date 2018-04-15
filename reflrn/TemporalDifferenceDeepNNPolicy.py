import logging
import random
import unittest
from pathlib import Path
from random import randint

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.EvaluationException import EvaluationException
from reflrn.Policy import Policy
from reflrn.State import State
from reflrn.TemporalDifferenceDeepNNPolicyPersistance import TemporalDifferenceDeepNNPolicyPersistance


#
# This depends on the saved Q-Values from TemporalDifferencePolicy. It trains itself on those Q Values
# and then implements greedy action based on the output of the Deep NN. Where the Deep NN is trained as a
# to approximate the function of the Q Values given a state for a given agent.
#


class TemporalDifferenceDeepNNPolicy(Policy):
    __epochs = 50
    __batch_size = 10

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self, lg: logging):
        self.__lg = lg
        self.__model = None
        return

    #
    # Define the model that will approximate the Q Value function.
    # X: 1 by 9 : State Of the Board.
    # Y: 1 by 9 : Q Values for the 9 possible actions.
    #
    @classmethod
    def __set_up_model(cls) -> keras.models.Sequential:

        model = Sequential()
        model.add(Dense(1000, input_dim=9, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(9))
        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

        return model

    #
    # Train the model given a set of State/Q Values saved from the TemporalDifferencePolicy Class. The model's
    # ability to predict the action will be no better then the ability of the Q Values. This is simply an exercise
    # in using a Deep NN as an interim step to understanding the full actor/critic policy.
    #
    # The given file name is the name of a Q Value dump file.
    #
    def train(self, qval_file_name: str, model_file_name: str, load_model_if_present: bool = False) -> float:
        #
        # Load the Q Values and States as learned by the TemporalDifferencePolicy class.
        #
        x, y = TemporalDifferenceDeepNNPolicyPersistance(lg=self.__lg).load_state_qval_as_xy(filename=qval_file_name)

        self.__model = None
        if load_model_if_present:
            if Path(model_file_name).is_file():
                self.__model = keras.models.load_model(model_file_name)

        if self.__model is None:
            self.__model = self.__set_up_model()
            self.__model.fit(x, y, epochs=self.__epochs, batch_size=self.__batch_size)
            self.__model.save(model_file_name)

        scores = self.__model.evaluate(x, y)
        self.__lg.debug("%s: %.2f%%" % (self.__model.metrics_names[1], scores[1] * 100))
        return scores[1] * 100

    #
    # We do not update the policy, we just train the model once at the outset via the train() method.
    # This is **not** a Critic / Actor Pattern
    #
    def update_policy(self,
                      agent_name: str,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool):
        return

    #
    # Greedy action; request human user to input action.
    #
    def greedy_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:

        qvs = None
        if self.__model is not None:
            x = TemporalDifferenceDeepNNPolicyPersistance.state_as_str_to_numpy_array(state.state_as_string())
            qvs = self.__model.predict(np.array([x]))[0]
            self.__lg.debug("Predict Y:= " + str(qvs))
        else:
            raise EvaluationException("No (Keras) Model Loaded with which to predict Q Values")

        ou = np.max(qvs)
        greedy_actions = list()
        for i in range(0, len(qvs)):
            if qvs[i] == ou:
                if i in possible_actions:
                    greedy_actions.append(int(i))
        if len(greedy_actions) == 0:
            raise EvaluationException("Model did not predict a Q Values related to a possible action")

        return greedy_actions[randint(0, len(greedy_actions) - 1)]

    #

    # Save the Keras Deep NN
    #
    def save(self, filename: str = None):
        # No save as model is not updated during runs
        return

    #
    # Load the Keras Deep NN
    #
    def load(self, filename: str = None):
        if Path(filename).is_file():
            self.__model = keras.models.load_model(filename)
        return


# ********************
# *** UNIT TESTING ***
# ********************


class TestTemporalDifferenceDeepNNPolicy(unittest.TestCase):
    __qval_file = 'qvn_dump.pb'
    __model_file = 'model.keras'
    __lg = None

    @classmethod
    def setUpClass(cls):
        random.seed(42)
        np.random.seed(42)
        cls.__lg = EnvironmentLogging("TestTemporalDifferenceDeepNNPolicy",
                                      "TestTemporalDifferenceDeepNNPolicy.log",
                                      logging.INFO).get_logger()

    def test_training(self):
        tddnnp = TemporalDifferenceDeepNNPolicy(lg=self.__lg)
        accuracy = tddnnp.train(self.__qval_file, self.__model_file, load_model_if_present=True)
        self.assertGreaterEqual(accuracy, float(90))
        return


#
# Execute the tests.
#


if __name__ == "__main__":
    tests = TestTemporalDifferenceDeepNNPolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
