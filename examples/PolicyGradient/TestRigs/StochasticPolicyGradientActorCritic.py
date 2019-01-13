import random
from collections import deque
from typing import Tuple

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from reflrn.SimpleLearningRate import SimpleLearningRate


#
# This is the environment that returns rewards. The reward profile is -x2 where the state is
# x between -1 and 1 on 0.05 step intervals. This means the optimal action to maximise reward
# is to always move such that x approaches 0. The actions are 0 = move right on x axis and 1
# move left on x axis. -0.05, 0 & 0.05 all return zero reward such that the agent should tend
# to move such that it stays in this region.
#
# The environment is episodic such that if actor moves below -1 or above 1 the episode restarts.
#

class PbFunc:
    state_min = -1
    state_max = 1
    state_step = 0.05

    #
    # reset to start of an episode.
    #
    def __init__(self):
        self.state = None
        self.reset()

    #
    # Return float (state) as numpy array to be used a NN X input.
    #
    @classmethod
    def float_as_x(cls,
                   x: float):
        xs = np.array([x])
        return xs.reshape([1, xs.shape[0]])

    #
    # The actor has moved pass -1 or +1 and episode ends and agent is randomly placed at -1 or 1
    #
    def reset(self) -> np.array:
        if np.random.rand() >= 0.5:
            self.state = self.state_max
        else:
            self.state = self.state_min
        return np.array([self.state])

    #
    # state space is dimension 1 as the state is full represented by the x value (single float)
    #
    @classmethod
    def state_space_size(cls) -> int:
        return 1

    #
    # Action space is 2 - move left by 0.05 or move right by 0.05
    #
    @classmethod
    def num_actions(cls) -> int:
        return 2

    #
    # Reward is simple -x2 where x is the state. Except at 0.05, 0, -0.05 where reward is fixed at 0
    #
    @classmethod
    def reward(cls,
               st: float) -> float:
        if -0.05 <= st <= 0.05:
            return float(0)
        return -(st * st)

    #
    # Translate action into step left or right increment of x.
    #
    def step(self,
             actn: int) -> Tuple[np.array, float, bool]:
        if actn == 0:
            self.state += self.state_step
        elif actn == 1:
            self.state -= self.state_step
        else:
            raise RuntimeError("Action can only be value 0 or 1 so [" + str(actn) + "] is illegal")

        self.state = np.round(self.state, 3)
        dn = (self.state < self.state_min or self.state > self.state_max)

        return np.array([self.state]), self.reward(self.state), dn


#
# Two network actor / critic stochastic policy with the critic learning state q-values on a one step Bellman.
#
# Exploration is inherent as policy is stochastic
#
class PGAgent:

    def __init__(self, st_size, a_size):
        self.state_size = st_size
        self.action_size = a_size
        self.gamma = 0  # 0.99
        self.learning_rate = 0.001
        self.replay = deque(maxlen=1000)
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()

        self.actor_model.summary()

        self.critic_model.summary()
        qval_lr0 = float(1)
        self.qval_learning_rate = SimpleLearningRate(lr0=qval_lr0,
                                                     lrd=SimpleLearningRate.lr_decay_target(learning_rate_zero=qval_lr0,
                                                                                            target_step=1000,
                                                                                            target_learning_rate=0.01),
                                                     lr_min=0.01)

        self.state_dp = 3

        return

    #
    # Simple NN model with softmax learning the policy as probability distribution over actions.
    #
    def _build_actor_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    #
    # Simple NN model learning QValues by state.
    #
    def _build_critic_model(self):
        model = Sequential()
        model.add(Dense(25, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    #
    # Retain the episode state for critic training.
    #
    def remember(self,
                 state: np.array,
                 action,
                 r: float,
                 next_state: np.array) -> None:
        y = np.zeros([self.action_size])
        y[action] = 1  # One hot encode.
        self.replay.append([np.round(state, self.state_dp),
                            np.array(y).astype('float32'),
                            r,
                            np.round(next_state, 2)])
        return

    #
    # Act according to the current stochastic policy
    #
    def act(self, state) -> int:
        state = state.reshape([1, state.shape[0]])
        aprob = self.actor_model.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=aprob)[0]
        return action

    #
    # learning rate for the q-value update
    #
    def qval_lr(self,
                episode: int) -> float:
        return self.qval_learning_rate.learning_rate(episode)

    #
    # Train the critic to learn the action values. Bellman
    #
    def train_critic(self,
                     episode: int) -> None:
        batch_size = min(len(self.replay), 100)
        X = np.zeros(batch_size)
        Y = np.zeros((batch_size, self.action_size))
        samples = random.sample(list(self.replay), batch_size)
        i = 0
        for sample in samples:
            state, action_one_hot, reward, next_state = sample
            lr = self.qval_lr(episode)
            action_value_s = self.critic_model.predict(state, batch_size=1).flatten()
            action_value_ns = self.critic_model.predict(next_state, batch_size=1).flatten()
            qv_s = action_one_hot * action_value_s
            qv_ns = np.max(action_value_ns) * self.gamma
            av = (qv_s * (1 - lr)) + (lr * (reward + qv_ns))  # updated expectation of current state/action
            qv_u = (action_value_s * (1 - action_one_hot)) + (av * action_one_hot)
            X[i] = np.squeeze(state)
            Y[i] = np.squeeze(qv_u)
            i += 1
        self.critic_model.train_on_batch(X, Y)
        return

    #
    # Train the actor to learn the stochastic policy; the reward is the reward for the action
    # as predicted by the critic.
    #
    def train_actor(self) -> None:
        batch_size = min(len(self.replay), 100)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        samples = random.sample(list(self.replay), batch_size)
        i = 0
        for sample in samples:
            state, action_one_hot, reward, next_state = sample
            action_value_s = self.critic_model.predict(state, batch_size=1).flatten()
            r = action_one_hot * action_value_s
            X[i] = state
            Y[i] = r
            i += 1
        self.actor_model.train_on_batch(X, Y)
        return

    def load(self, name) -> None:
        self.actor_model.load_weights('actor' + name)
        self.critic_model.load_weights('critic' + name)

    def save(self, name) -> None:
        self.actor_model.save_weights('actor' + name)
        self.critic_model.save_weights('critic' + name)

    #
    # Simple debugger output - could be refactored into PBFunc env as it is more env specific ?
    #
    def print_progress(self,
                       ep: int,
                       elen: int,
                       e: PbFunc) -> None:
        res = str()
        ts = e.state_min
        while ts <= e.state_max:
            ss = PbFunc.float_as_x(ts)
            aprob = self.actor_model.predict(ss, batch_size=1).flatten()
            print("S: " + '{:+.2}'.format(float(ts)) + " [ " +
                  str(round(aprob[0] * 100, 2)) + "% , " +
                  str(round(aprob[1] * 100, 2)) + "%]")
            if aprob[0] > aprob[1]:
                res += '>'
            else:
                res += "<"
            if ts == 0:
                res += "|"
            ts += e.state_step
        print(str(ep) + "::" + str(elen) + "   " + res)
        return


class Test:
    @classmethod
    def run(cls):
        stp = 0.05
        env = PbFunc()
        state_size = env.state_space_size()
        action_size = env.num_actions()
        agent = PGAgent(state_size, action_size)
        PGAgent.gamma = 0
        for i in range(0, 5):
            j = float(-1)
            while j <= 1:
                agent.remember(np.array([j]), 0, +j, np.array([j + stp]))
                agent.remember(np.array([j]), 1, -j, np.array([j - stp]))
                j += stp
        for i in range(0, 1000):
            agent.train_critic(i)
            j = float(-1)
            print("-----")
            while j <= 1:
                print(str(j) + ' : ' + str(agent.critic_model.predict(np.array([j]), batch_size=1).flatten()))
                j += stp
            print("-----")
        return


#
# Test Rig Main Function.
#
class Main:

    @classmethod
    def run(cls):
        env = PbFunc()
        st = env.reset()
        episode = 0
        eln = 0

        state_size = env.state_space_size()
        action_size = env.num_actions()
        agent = PGAgent(state_size, action_size)
        while True:
            a = agent.act(st)
            next_state, reward, done = env.step(a)
            agent.remember(st, a, reward, next_state)
            st = next_state
            eln += 1

            if done or eln > 10000:
                if episode > 3:
                    agent.train_critic(episode)
                if episode > 1 and episode % 5 == 0:
                    agent.train_actor()
                    agent.print_progress(episode, eln, env)
                eln = 0
                st = env.reset()
                episode += 1


#
# Endless loop of agent acting / learning in the given env.
#
test = True
if __name__ == "__main__":
    if test:
        Test.run()
    else:
        Main().run()
