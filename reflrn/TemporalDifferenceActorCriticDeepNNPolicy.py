import logging
from pathlib import Path
from random import randint

import keras
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential

from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State
from reflrn.Interface.Model import Model
from reflrn.DequeReplayMemory import DequeReplayMemory
from reflrn.EvaluationException import EvaluationException
from reflrn.ModelParameters import ModelParameters


#
# This follows the TestGridActorCritic pattern.
#


class TemporalDifferenceActorCriticDeepNNPolicy(Policy):
    # Learning Parameters
    __n = 0  # number of learning events
    __learning_rate_0 = float(0.05)
    __discount_factor = float(0.8)
    __learning_rate_decay = float(0.001)

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self,
                 lg: logging,
                 replay_memory: DequeReplayMemory,
                 model_file_name: str,
                 num_actions: int,
                 model: Model,
                 model_parameters: ModelParameters = None,
                 load_model: bool = False):

        self.__lg = lg
        if load_model:
            self.load(model_file_name)
        else:
            self.__actor = self.create_new_model_instance()
            self.__critic = self.create_new_model_instance()
        self.__episodes_played = 0
        self.__model_file_name = model_file_name
        self.__critic_updates = 0
        self.__replay_memory = replay_memory
        self.__model = model
        if model_parameters is None:
            mp = ModelParameters()  # Defaults
        else:
            mp = model_parameters

        self.__num_actions = num_actions
        self.__epochs = mp.epochs
        self.__update_every_n_episodes = mp.update_every_n_episodes
        self.__sample_size = mp.sample_size
        self.__batch_size = mp.batch_size
        self.__replay_mem_size = mp.replay_mem_size
        self.__save_every_n_critic_updates = mp.save_every_n_critic_updates

        return

    #
    # Define the model that will approximate the Q Value function.
    # X & Y dimensions must match the num actions (= num q values per state)
    #
    def create_new_model_instance(self) -> keras.models.Sequential:
        return self.__model.new_model()

    #
    # Compile the Keras Model
    #
    def __compile_model(self, model):
        self.__model.compile()
        return

    #
    # Return the learning rate based on number of learning's to date
    #
    @classmethod
    def __learning_rate(cls):
        return cls.__learning_rate_0 / (1 + (cls.__n * cls.__learning_rate_decay))

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

        # Record the number of learning events.
        self.__n += 1

        # Track the SAR for critic training.
        self.__replay_memory.append_memory(state, next_state, action, reward, episode_complete)

        # Track the number of finished episodes, this drives the learning updates.
        if episode_complete:
            self.__episodes_played += 1

        # Every so often, we train the critic from memories and swap it in as the new actor.
        if self.__replay_memory.len() > self.__sample_size and (
                self.__episodes_played % self.__update_every_n_episodes) == 0:
            memories = self.get_random_memories()
            x, y = self.random_memories_to_training_xy(memories)
            self.train_critic_and_update_actor(x, y)

    #
    # Extract a random set of memories equal to the defined sample size.
    #
    def get_random_memories(self):
        return self.__replay_memory.get_random_memories(self.__sample_size)

    #
    # Convert a given set of memories to x,y training inputs. As part of this we apply the temporal difference
    # updates to the q values predicted by the actor.
    #
    def random_memories_to_training_xy(self, memories):
        lnm = len(memories[0])
        x = np.empty((lnm, 9))
        y = np.empty((lnm, 9))
        for i in range(0, lnm):
            state = memories[DequeReplayMemory.mem_state][i]
            next_state = memories[DequeReplayMemory.mem_next_state][i]
            action = memories[DequeReplayMemory.mem_action][i]
            reward = memories[DequeReplayMemory.mem_reward][i]

            # Extract states as numpy arrays (in form needed for the NN as input X)

            s = self.state_as_str_to_numpy_array(state.state_as_string())
            ns = self.state_as_str_to_numpy_array(next_state.state_as_string())

            lr = self.__learning_rate()

            qvns = self.__actor.predict(np.array([ns]))[0]
            mxq = np.max(qvns)
            qvp = self.__discount_factor * mxq * lr

            qv = self.__actor.predict(np.array([s]))[0]
            qv[action] = (qv[action] * (1 - lr)) + (lr * reward) + qvp
            mn = np.min(qv)
            mx = np.max(qv)
            if mx - mn != 0:
                qv = (qv - mn) / (mx - mn)

            x[i] = s
            y[i] = qv

        return x, y

    #
    # Train the model given a set of State/Q Values saved from the TemporalDifferencePolicy Class. The model's
    # ability to predict the action will be no better then the ability of the Q Values. This is simply an exercise
    # in using a Deep NN as an interim step to understanding the full actor/critic policy.
    #
    # The given file name is the name of a Q Value dump file.
    #
    def train_critic_and_update_actor(self, x, y):

        self.__critic.fit(x, y, epochs=self.__epochs, batch_size=self.__batch_size, verbose=0)
        scores = self.__critic.evaluate(x, y)
        self.__lg.info("Critic Accuracy %s: %.2f%%" % (self.__critic.metrics_names[1], scores[1] * 100))

        self.__critic_updates += 1
        if self.__critic_updates % self.__save_every_n_critic_updates == 0:
            self.__critic.save(self.__model_file_name)
            self.__lg.info("Critic Saved")
        self.__actor = keras.models.clone_model(self.__critic)  # Make the actor = to (newly retrained) Critic
        self.__compile_model(self.__actor)
        return

    #
    # Convert a state string to numpy vector
    #
    @classmethod
    def state_as_str_to_numpy_array(cls, xs: str) -> np.array:
        xl = list()
        s = 1
        for c in xs:
            if c == '-':
                s = -1
            else:
                xl.append(float(c) * float(s))
                s = 1
        return np.asarray(xl, dtype=np.float32)

    #
    # Greedy action; The current version of the Actor predicts the Q Values.
    #
    def select_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:

        qvs = None
        if self.__actor is not None:
            qvs = self.__actor.predict(np.array([self.state_as_str_to_numpy_array(state.state_as_string())]))[0]
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
    # Save, not relevant for this Policy type
    #
    def save(self, filename: str = None):
        # Save is done as part of policy update, not on the environment call back
        return

    #
    # Load the last critic Keras Deep NN
    #
    def load(self, filename: str = None):
        if Path(filename).is_file():
            self.__actor = keras.models.load_model(filename)
            self.__critic = keras.models.clone_model(self.__actor)
            self.__compile_model(self.__critic)
            self.__lg.info("Critic & Actor Loaded from file: " + filename)
        else:
            self.__actor = self.create_new_model_instance()
            self.__critic = self.create_new_model_instance()
            self.__lg.info("Critic & Actor not loaded as file not found: " + filename)
        return
