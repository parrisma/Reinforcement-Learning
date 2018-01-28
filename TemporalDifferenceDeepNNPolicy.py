import unittest
import logging
import keras
from keras.models import Sequential
from keras.layers import Dense
from Persistance import Persistance
import numpy as np
from pathlib import Path
from Policy import Policy
from State import State
from typing import Tuple

#
# This depends on the saved Q-Values from TemporalDifferencePolicy. It trains itself on those Q Values
# and then implements greedy action based on the output of the Deep NN. Where the Deep NN is trained as a
# to approximate the function of the Q Values given a state for a given agent.
#


class TemporalDifferenceDeepNNPolicy(Policy):

    __model = None

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self, agent_name: str, lg: logging):
        self.__lg = lg
        self.__agent_name = agent_name
        return

    #
    # Define the model that will approximate the Q Value function.
    # X: 1 by 10 : Actor Id + State Of the Board.
    # Y: 1 by 9 : Q Values for the 9 possible actions.
    #
    @classmethod
    def __model(cls) -> keras.models.Sequential:

        model = Sequential()
        model.add(Dense(1000, input_dim=10, activation='relu'))
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
    def train(self, file_name: str):
        return

    #
    # We do not update the policy, we just train the model once at the outset via the train() method.
    # This is **not** a Critic / Actor Pattern
    #
    def update_policy(self, agent_name: str, prev_state: State, prev_action: int, state: State, action: int, reward: float):
        return

    #
    # Greedy action; request human user to input action.
    #
    def greedy_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:
        mv = None
        return mv

    #
    # Save the Keras Deep NN
    #
    def save(self, filename: str=None):
        return

    #
    # Load the Keras Deep NN
    #
    def load(self, filename: str)-> Tuple[dict, int, np.float, np.float, np.float]:
        return

# ********************
# *** UNIT TESTING ***
# ********************


class TestTemporalDifferenceDeepNNPolicy(unittest.TestCase):

        def test_training(self):
            #self.assertEqual(se.select_action(possible_actions), expected_action)
            return

#
# Execute the tests.
#


if __name__ == "__main__":
    tests = TestTemporalDifferenceDeepNNPolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
