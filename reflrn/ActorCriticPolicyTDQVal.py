import os
from copy import deepcopy
from typing import Tuple

import numpy as np

from examples.tictactoe.RenderQValuesAsStr import RenderQValues
from reflrn.ActorCriticPolicyTelemetry import ActorCriticPolicyTelemetry
from reflrn.DictReplayMemory import DictReplayMemory
from reflrn.GeneralModelParams import GeneralModelParams
from reflrn.Interface.Environment import Environment
from reflrn.Interface.ModelParams import ModelParams
from reflrn.Interface.NeuralNetwork import NeuralNetwork
from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State
from reflrn.QValNNModel import QValNNModel
from reflrn.SimpleLearningRate import SimpleLearningRate


#
# This depends on the saved Q-Values from TemporalDifferencePolicy. It trains itself on those Q Values
# and then implements greedy action based on the output of the Deep NN. Where the Deep NN is trained as a
# to approximate the function of the Q Values given a curr_coords for a given agent.
#


class ActorCriticPolicyTDQVal(Policy):
    __replay_mem_size = 1000

    __static_test_action_list = None  # Static list of actions for Policy testing
    __next_test_action = 0

    def __init__(self,
                 lg,
                 network: NeuralNetwork,
                 policy_params: GeneralModelParams = None,
                 env: Environment = None):

        self.env = env  # If Env not passed, then must be bound via link_to_env() method.
        self.lg = lg
        self.episode = 1

        self.input_dim = network.input_dimension()
        self.num_actions = network.output_dimension()
        self.output_dim = self.num_actions

        pp = self._default_policy_params()
        if policy_params is not None:
            pp.override_parameters(list(policy_params))
        self.learning_rate = SimpleLearningRate(lr0=pp.get_parameter(ModelParams.learning_rate_0),
                                                lrd=pp.get_parameter(ModelParams.learning_rate_decay),
                                                lr_min=pp.get_parameter(ModelParams.learning_rate_min))
        self.epsilon = pp.get_parameter(ModelParams.epsilon)  # exploration factor.
        self.epsilon_decay = pp.get_parameter(ModelParams.epsilon_decay)
        self.gamma = pp.get_parameter(ModelParams.gamma)  # Discount Factor Applied to reward
        self.verbose = pp.get_parameter(ModelParams.verbose)  # Verbose output from model while training.
        self.train_every = pp.get_parameter(ModelParams.train_every)
        self.save_every = 1000
        self.num_states = pp.get_parameter(ModelParams.num_states)

        self.__training = True  # by default we train actor/critic as we take actions
        self.__train_invocations = 0
        self.__trained = False
        self.__critic_train_count = 0

        self.__explore = True

        #
        # Replay memory needed to model a stationary target.
        #
        self.__replay_memory = DictReplayMemory(lg, self.__replay_mem_size)

        #
        # Create the actor / critic NN models that will work as the function approximations for Q Vals.
        #
        self.actor_model = QValNNModel(model_name="Actor",
                                       input_dimension=self.input_dim,
                                       num_actions=self.num_actions,
                                       network=network,
                                       lg=self.lg,
                                       model_params=self._model_params()
                                       )

        self.critic_model = QValNNModel(model_name="Critic",
                                        input_dimension=self.input_dim,
                                        num_actions=self.num_actions,
                                        network=network,
                                        lg=self.lg,
                                        model_params=self._model_params()
                                        )
        self.__explain = None
        self.explain = False

        self.__telemetry = ActorCriticPolicyTelemetry()

        return

    #
    # Make a note of which environment policy is linked to.
    #
    def link_to_env(self, env: Environment) -> None:
        if self.env is not None:
            raise Policy.PolicyAlreadyLinkedToEnvironment("Policy already linked to an environment !")
        self.env = env
        return

    #
    # Indirection for getting policy so exception can be raised if policy was not linked.
    #
    def _env(self) -> Environment:
        if self.env is None:
            raise ActorCriticPolicyTDQVal.NoEnvironmentHasBeenLinkedToPolicy(
                "Policy must be linked to an Environment !")
        return self.env

    #
    # Train (model) on incoming events.
    #
    def set_training_on(self) -> None:
        self.__training = True

    #
    # Ignore incoming events from (model) training perspective.
    #
    def set_training_off(self) -> None:
        self.__training = False

    #
    # Reset policy to be ready for start of new episode
    #
    def __episode_complete_reset(self) -> None:

        self.episode += 1

        if self.__static_test_action_list is not None:
            self.__next_test_action = 0

        return

    #
    # Update the policy with respect to the state / action transition that has occurred in the environment to which
    # this policy is linked and
    #
    def update_policy(self,
                      agent_name: str,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool) -> None:

        if episode_complete:
            self.__episode_complete_reset()

        self.__replay_memory.append_memory(state,
                                           next_state,
                                           action,
                                           reward,
                                           episode_complete)
        self._train(self.train_every)

    #
    # Select a random allowable action in the current state
    #
    def __random_action(self,
                        state: State) -> int:
        actions_allowed_in_current_state = self._env().actions(state)
        actn = (np.random.choice(actions_allowed_in_current_state, 1))[0]
        return actn

    #
    # Q-Value prediction of critic
    #
    def __critic_prediction(self,
                            state: State):
        return (self.critic_model.predict(state.state_as_array().reshape(1, self.input_dim)))[0]

    #
    # Predict an action based on current policy.
    #
    def __predict_action(self,
                         state: State) -> int:
        actions_allowed_in_current_state = self._env().actions(state)
        actions_not_allowed_in_current_state = self.actions_taken(actions_allowed_in_current_state)
        qvals = self.__critic_prediction(state)
        if len(actions_not_allowed_in_current_state) > 0:
            qvals[actions_not_allowed_in_current_state] = np.finfo(np.float).min
        if self.explain:
            print(RenderQValues.render_simple(qvals))
            print(self.__telemetry.state_observation_telemetry(state.state_as_array()))
        actn = self._env().actions()[int(np.argmax(qvals))]
        return actn

    #
    # Select a static test action
    #
    def __select_static_test_action(self):
        if self.__static_test_action_list is None:
            raise NotImplementedError("No static test action list was defined")
        if self.__next_test_action >= len(self.__static_test_action_list):
            self.__next_test_action = 0
        self.__next_test_action += 1
        return self.__static_test_action_list[self.__next_test_action - 1]  # yes -1

    #
    # Based on exploration policy and current critic model, either take a random action
    # based on setting of epsilon or use the current critical model to predict a greedy
    # action based on highest expected return.
    #
    def select_action(self, agent_name: str,
                      state: State,
                      possible_actions: [int]) -> int:

        if self.__static_test_action_list is not None:
            return self.__select_static_test_action()

        exp = np.random.rand()
        if exp > self.__epsilon():
            # Random Exploration
            actn = self.__random_action(state)
            if actn not in self._env().actions(state):
                raise self.IllegalActionPrediction("R: Action prediction has given an action not legal for given state")
        else:
            # Greedy exploration
            actn = self.__predict_action(state)
            if actn not in self._env().actions(state):
                raise self.IllegalActionPrediction("P: Action prediction has given an action not legal for given state")
        return actn

    #
    # Make a prediction based only on QValues of given state
    #
    def greedy_action(self,
                      state: State) -> int:

        if self.__static_test_action_list is not None:
            return self.__select_static_test_action()

        qvals = (self.critic_model.predict(state.state_as_array().reshape(1, self.input_dim)))[0]
        actn = np.argmax(qvals)
        return actn

    def actions_taken(self,
                      actions_remaining: np.ndarray) -> np.ndarray:
        actns = list()
        for a in self._env().actions():
            if a not in actions_remaining:
                actns.append(a)
        return np.asarray(actns)

    #
    # Persist the current Policy
    #
    def save(self,
             filename: str = None
             ) -> None:
        try:
            # Save the telemetry data & Model Data
            self.__telemetry.save(filename + '.tlm')
            self.actor_model.save(filename + '_actor.pb')
            self.critic_model.save(filename + '_critic.pb')
        except Exception as exc:  # report but ignore error
            err = "Failed to save ActorCriticPolicy to file [" + filename + ": " + str(exc)
            self.lg.error(err)
        return

    #
    # Load the Policy from a persisted copy
    #
    def load(self, filename: str = None) -> None:
        if filename is not None and len(filename) > 0:
            if os.path.isfile(filename + '_actor.pb') and os.path.isfile(filename + '_critic.pb'):
                self.actor_model.load(filename + '_actor.pb')
                self.critic_model.load(filename + '_critic.pb')
            else:
                raise ValueError("Cannot load model files given filename root :" + filename)
        return

    #
    # Sufficient experience to start training ?
    #
    def _sufficient_experience_to_start_training(self) -> bool:
        return self.__replay_memory.len() > min(100, max(1, int(self.num_states * 0.1)))

    #
    # Train the critic and then update the actor
    #
    # Only train the critic every train_every invocations
    # Only update the actor every update_every episodes
    #
    def _train(self,
               train_every: int = 50,
               update_every: int = 5) -> None:

        if not self.__training:
            return

        self.__train_invocations += 1

        if self._sufficient_experience_to_start_training():
            if self.__train_invocations % train_every == 0:
                self._train_critic()
                self.__train_invocations = 0
                self.__trained = True
                self.__critic_train_count += 1

            if self.__critic_train_count % update_every == 0 and self.__trained:
                self._update_actor_from_critic()
                self.__trained = False  # No need to update unless model has trained since last update

            if self.__critic_train_count % self.save_every == 0:
                self.save("ActorCriticPolicy1")
        return

    #
    # Get actor prediction, if actor is not able to predict, predict random
    #
    def _actor_prediction(self,
                          curr_state: State) -> np.ndarray:
        st = np.array(curr_state.state_as_array()).reshape((1, self.input_dim))  # Shape needed for NN
        p = self.actor_model.predict(st)[0]  # Can predict even if model is not trained, just predicts random.
        return p

    #
    # The projected reward if for taking greedy actions until the end of the episode.
    # However, if this is a terminal state by definition the best possible reward is always zero.
    #
    def _next_state_qval_prediction(self,
                                    new_state: State,
                                    done: bool) -> float:
        qvp = 0
        if not (self._env().episode_complete(new_state) or done):
            qvn = self._actor_prediction(curr_state=new_state)
            allowable_actions = self._env().actions(new_state)
            qvp = np.max(qvn[allowable_actions])  # Greedy reward from current state
        return np.float(qvp)

    #
    # What is the q_value model prediction given current state S, by definition of this is a
    # terminal state all rewards are zero.
    #
    def _curr_state_qval_prediction(self,
                                    curr_state: State,
                                    done: bool) -> np.ndarray:
        qvs = np.zeros(self.num_actions)
        if not done:
            qvs = self._actor_prediction(curr_state=curr_state)
        return qvs

    # Get a random set of samples from the given QValues to select_action as a test or training
    # batch for the model.
    #
    def _get_sample_batch(self) -> Tuple[np.ndarray, np.ndarray]:

        batch_size = self._model_params().get_parameter(ModelParams.batch_size)
        samples = self.__replay_memory.get_random_memories(batch_size)

        x = np.zeros((batch_size, self.input_dim))
        y = np.zeros((batch_size, self.num_actions))
        i = 0
        for sample in samples:
            _, cur_state, new_state, action, reward, done = sample
            lr = self.learning_rate.learning_rate(self.episode)
            qvp = self._next_state_qval_prediction(new_state, done)
            qvs = self._curr_state_qval_prediction(cur_state, done)

            qv = qvs[action]
            qv = (qv * (1 - lr)) + (lr * (reward + qvp))  # updated expectation of current state/action
            # qv = reward + (self.gamma * qvp)  # updated expectation of current state/action

            qvs[action] = qv

            x[i] = (cur_state.state_as_array()).reshape(1, self.input_dim)
            y[i] = qvs
            i += 1

        return x, y

    #
    # Actor is not trained, but instead clones the trainable parameters from the critic
    # after every n times the critic is trained on a replay memory batch.
    #
    def _update_actor_from_critic(self) -> None:
        self.actor_model.clone_weights(self.critic_model)
        self.lg.debug("Update Actor From Critic")
        return

    #
    # Extract a random sample from the replay memory, calculate the QVals and train
    # the critical deep NN model.
    #
    def _train_critic(self) -> bool:
        trained = False
        rw, cl = self._get_sample_batch()
        if rw is not None:
            self.critic_model.train(rw, cl)
            trained = True
            self.lg.debug("Critic Trained")
            self._model_loss(self.critic_model)
            self.update_telemetry_for_training_batch(rw)
        return trained

    #
    # Establish the model loss
    #
    def _model_loss(self,
                    mdl) -> float:
        rw, cl = self._get_sample_batch()
        loss = None
        if rw is not None:
            scores = mdl.evaluate(rw, cl)
            if type(scores) == list:
                loss = scores[0]
            else:
                loss = scores
            print("Loss: {}".format(loss))
        return loss

    #
    # Indirection to exploration factor so we can turn exploration on/off so when we play
    # we only play greedy actions.
    #
    def __epsilon(self) -> float:
        if not self.__explore:
            return float(1)
        return self.epsilon

    def exploration_on(self) -> None:
        self.__explore = True

    def exploration_off(self) -> None:
        self.__explore = False

    #
    # The parameters needed by the Keras Model
    #
    def _model_params(self) -> ModelParams:
        mp = GeneralModelParams([[ModelParams.learning_rate_0, 0.001],
                                 [ModelParams.learning_rate_min, 0.001],
                                 [ModelParams.batch_size, 32],
                                 [ModelParams.verbose, self.verbose]
                                 ]
                                )
        return mp

    #
    # The parameters needed by the iterative training process.
    #
    @classmethod
    def _default_policy_params(cls) -> ModelParams:
        pp = GeneralModelParams([[ModelParams.learning_rate_0, float(1)],
                                 [ModelParams.learning_rate_decay, float(0.02)],
                                 [ModelParams.epsilon, float(0.8)],
                                 [ModelParams.epsilon_decay, float(0.9995)],
                                 [ModelParams.gamma, float(0.8)],
                                 [ModelParams.verbose, int(0)],
                                 [ModelParams.train_every, int(100)],
                                 [ModelParams.num_states, int(1)]  # Env Specific - should be overridden
                                 ],
                                )
        return pp

    # Illegal action Prediction
    #
    class IllegalActionPrediction(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    # Policy was not linked to an environment before it was used..
    #
    class NoEnvironmentHasBeenLinkedToPolicy(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    #
    # Update policy telemetry for training batch
    #
    def update_telemetry_for_training_batch(self,
                                            x: np.ndarray):
        for i in x:
            self.__telemetry.update_state_telemetry(i)
        return

    #
    # Generate debug details when predicting actions.
    #
    @property
    def explain(self) -> bool:
        return self.__explain

    @explain.setter
    def explain(self,
                value: bool) -> None:
        if value is None:
            raise ValueError()
        if type(value) != bool:
            raise TypeError("explain property is type bool cannot not [" + type(value).__name__ + "]")
        self.__explain = value
        self.actor_model.explain = self.__explain
        self.critic_model.explain = self.__explain

    #
    # Define static list of (rolling) test actions
    #
    @property
    def static_test_action_list(self) -> list:
        return self.__static_test_action_list

    @static_test_action_list.setter
    def static_test_action_list(self,
                                value: list) -> None:
        if value is None:
            raise ValueError()
        if len(value) == 0:
            raise ValueError("Test action list must be list of one or more actions")
        if not (type(value) == list or type(value) == tuple):
            raise TypeError("test action list property must be of type list/tuple not [" + type(
                value).__name__ + "]")

        self.__static_test_action_list = deepcopy(value)
        self.__next_test_action = 0
