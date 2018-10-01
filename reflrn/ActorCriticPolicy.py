from typing import List

import numpy as np

from reflrn.DequeReplayMemory import DequeReplayMemory
from reflrn.Interface.Environment import Environment
from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State
from reflrn.QValNNModel import QValNNModel


#
# This depends on the saved Q-Values from TemporalDifferencePolicy. It trains itself on those Q Values
# and then implements greedy action based on the output of the Deep NN. Where the Deep NN is trained as a
# to approximate the function of the Q Values given a curr_coords for a given agent.
#


class ActorCriticPolicy(Policy):
    __rows = 3
    __cols = 3
    __num_actions = 9
    __replay_mem_size = 1000

    def __init__(self,
                 lg,
                 env: Environment = None):

        self.env = env  # If Env not passed, then must be bound via link_to_env() method.

        self.lg = lg

        self.episode = 1

        self.batch_size = 32
        self.input_dim = ActorCriticPolicy.__rows * ActorCriticPolicy.__cols
        self.num_actions = ActorCriticPolicy.__num_actions
        self.output_dim = self.num_actions

        self.learning_rate_0 = float(1.0)
        self.learning_rate_decay = float(0.02)
        self.epsilon = 0.8  # exploration factor.
        self.epsilon_decay = .9995
        self.gamma = .8  # Discount Factor Applied to reward

        self.__training = True  # by default we train actor/critic as we take actions

        #
        # Replay memory needed to model a stationary target.
        #
        self.__replay_memory = DequeReplayMemory(lg, self.__replay_mem_size)

        self.actor_model = QValNNModel(model_name="Actor",
                                       input_dimension=self.input_dim,
                                       num_actions=self.num_actions,
                                       lg=self.lg,
                                       batch_size=self.batch_size,
                                       lr_0=0.005,
                                       lr_min=0.001
                                       )

        self.critic_model = QValNNModel(model_name="Critic",
                                        input_dimension=self.input_dim,
                                        num_actions=self.num_actions,
                                        lg=self.lg,
                                        batch_size=self.batch_size,
                                        lr_0=0.005,
                                        lr_min=0.001
                                        )

    #
    # Make a note of which environment policy is linked to.
    #
    def link_to_env(self, env: Environment) -> None:
        if self.env is not None:
            raise ActorCriticPolicy.PolicyAlreadyLinkedToEnvironment("Policy already linked to an environment !")
        self.env = env
        return

    #
    # Indirection for getting policy so exception can be raised if policy was not linked.
    #
    def _env(self) -> Environment:
        if self.env is None:
            raise ActorCriticPolicy.NoEnvironmentHasBeenLinkedToPolicy("Policy must be linked to an Environment !")
        return self.env

    #
    # Train (model) on incoming events.
    #
    def set_training_on(self):
        self.__training = True

    #
    # Ignore incoming events from (model) training perspective.
    #
    def set_training_off(self):
        self.__training = False

    def update_policy(self, agent_name: str, state: State, next_state: State, action: int, reward: float,
                      episode_complete: bool) -> None:

        if episode_complete:
            self.episode += 1

        self.__replay_memory.append_memory(state,
                                           next_state,
                                           action,
                                           reward,
                                           episode_complete)
        self._train()

    #
    # Based on exploration policy and current critic model, either take a random action
    # based on setting of epsilon or use the current critical model to predict a greedy
    # action based on highest expected return.
    #
    def select_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:
        actions_allowed_in_current_state = self._env().actions(state)
        actions_not_allowed_in_current_state = self.actions_taken(actions_allowed_in_current_state)
        actn = None
        exp = np.random.rand()
        if exp > self.epsilon:
            # Random Exploration
            actn = (np.random.choice(actions_allowed_in_current_state, 1))[0]
            if actn not in self._env().actions(state):
                print("Bad Actn")
        else:
            # Greedy exploration
            qvals = (self.critic_model.predict(state.state_as_array().reshape(1, 9)))[0]
            if len(actions_not_allowed_in_current_state) > 0:
                qvals[actions_not_allowed_in_current_state] = np.finfo(np.float).min
            actn = np.argmax(qvals)
            if actn not in self._env().actions(state):
                print("Bad Actn")
        return actn

    def actions_taken(self,
                      actions_remaining: np.ndarray) -> np.ndarray:
        l = list()
        for a in self._env().actions():
            if a not in actions_remaining:
                l.append(a)
        return np.asarray(l)

    def save(self, filename: str = None) -> None:
        pass

    def load(self, filename: str = None):
        pass

    #
    # Update actor every 5 episodes (goal states reached)
    #
    def _train(self):
        if not self.__training:
            return

        if self.__replay_memory.len() > 100:  # don't start learning until we have reasonable nom of memories
            if self._train_critic():
                if self.episode % 10 == 0:
                    self._update_actor_from_critic()
        return

    #
    # Return the learning rate based on number of learning's to date
    #
    def learning_rate(self):
        return self.learning_rate_0 / (1 + (self.episode * self.learning_rate_decay))

    #
    # Get actor prediction, if actor is not able to predict, predict random
    #
    def _actor_prediction(self,
                          curr_state: State):
        st = np.array(curr_state.state()).reshape((1, self.__num_actions))  # Shape needed for NN
        p = self.actor_model.predict(st)[0]  # Can predict even if model is not trained, just predicts random.
        return p

    #
    # What is the optimal QVal prediction for next curr_coords S'. Return zero if next curr_coords
    # is the end of the episode.
    #
    def _next_state_qval_prediction(self,
                                    new_state: State) -> float:
        qvp = 0
        if not self._env().episode_complete(new_state):
            qvn = self._actor_prediction(curr_state=new_state)
            allowable_actions = self._env().actions(new_state)
            qvp = self.gamma * np.max(qvn[allowable_actions])  # Discounted max return from next curr_coords
        return np.float(qvp)

    #
    # What is the qvalue (optimal) prediction given current curr_coords (state S)
    #
    def _curr_state_qval_prediction(self,
                                    curr_state: State,
                                    done: bool) -> List[np.float]:
        qvs = np.zeros(self.num_actions)
        if not done:
            qvs = self._actor_prediction(curr_state=curr_state)
        return qvs

    # Get a random set of samples from the given QValues to select_action as a training
    # batch for the model.
    #
    def _get_sample_batch(self):

        samples = self.__replay_memory.get_random_memories(self.batch_size)

        x = np.zeros((self.batch_size, self.input_dim))
        y = np.zeros((self.batch_size, self.num_actions))
        i = 0
        for sample in samples:
            cur_state, new_state, action, reward, done = sample
            lr = self.learning_rate()
            qvp = self._next_state_qval_prediction(new_state)
            qvs = self._curr_state_qval_prediction(cur_state, done)

            qv = qvs[action]
            qv = (qv * (1 - lr)) + (lr * (reward + qvp))  # updated expectation of current curr_coords/action
            qvs[action] = qv

            x[i] = cur_state
            y[i] = qvs
            i += 1

        return x, y

    #
    # Actor is not trained, but instead clones the trainable parameters from the critic
    # after every n times the critic is trained on a replay memory batch.
    #
    def _update_actor_from_critic(self):
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
        return trained

    # Can only link to one environment in lifetime of policy.
    #
    class PolicyAlreadyLinkedToEnvironment(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    # Policy was not linked to an environment before it was used..
    #
    class NoEnvironmentHasBeenLinkedToPolicy(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)
