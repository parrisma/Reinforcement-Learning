import math
import random
from collections import deque
from typing import Tuple

import numpy as np
from keras.initializers import RandomUniform
from keras.initializers import Zeros
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam

from examples.PolicyGradient.TestRigs.Interface.RewardFunction2D import RewardFunction2D
from examples.PolicyGradient.TestRigs.RewardFunctions.LocalMaximaRewardFunction2D import LocalMaximaRewardFunction2D
from examples.PolicyGradient.TestRigs.Visualise2D import Visualise2D
from reflrn.SimpleLearningRate import SimpleLearningRate


#
# Two network actor / critic stochastic policy with the critic learning state q-values on a one step Bellman.
#
# Exploration is inherent as policy is stochastic
#
class PolicyGradientAgent2D:
    __fig1 = None
    __fig2 = None
    __plot_pause = 0.0001
    __seed = 42

    def __init__(self,
                 reward_function_2d: RewardFunction2D,
                 st_size,
                 a_size,
                 num_states):
        self.env = reward_function_2d
        self.state_size = st_size
        self.action_size = a_size
        self.num_states = num_states
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.replay = deque(maxlen=2500)
        self.replay_kl_factor = 0.0
        self.kl_update = 0
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()

        self.actor_model.summary()

        self.critic_model.summary()
        qval_lr0 = float(1)
        self.qval_learning_rate = SimpleLearningRate(lr0=qval_lr0,
                                                     lrd=SimpleLearningRate.lr_decay_target(learning_rate_zero=qval_lr0,
                                                                                            target_step=5000,
                                                                                            target_learning_rate=0.01),
                                                     lr_min=0.01)

        self.state_dp = 5
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.critic_acc_history = []
        self.actor_acc_history = []
        self.actor_exploration_history = []

        self.visual = Visualise2D()
        self.visual.show()

        return

    def visualise(self) -> Visualise2D:
        return self.visual

    def replay_kl(self):
        dl = 0
        sd = dict()
        for s in self.replay:
            state, _, _, _ = s
            sas = np.array2string(state, separator=';')
            if sas in sd:
                sd[sas] += 1
            else:
                sd[sas] = 1
            dl += 1
        if dl < 2:
            return 0
        qx = ((dl / len(sd)) / dl)
        kln = math.log(1.0 / qx)
        kls = 0.0
        u = 0.0
        c = 0
        for k, v in sd.items():
            px = v / dl
            u += px * math.log(max(px, 1e-12) / max(qx, 1e-12))
            if u > 0:
                kls += u
                c += 1
            # print('k:{:d} v:{:d} px:{:f} qx:{:f} u:{:f} kls:{:f}'.format(k, v, px, qx, u, kls))
        klp = (kls / c) / kln
        return klp

    #
    # Simple NN model with softmax learning the policy as probability distribution over actions.
    #
    def _build_actor_model(self):
        ki = RandomUniform(minval=-0.05, maxval=0.05, seed=self.__seed)
        bi = Zeros()
        model = Sequential()
        model.add(Dense(800, input_dim=self.state_size, activation='relu', kernel_initializer=ki, bias_initializer=bi))
        model.add(Dropout(0.1))
        model.add(Dense(400, activation='relu', kernel_initializer=ki, bias_initializer=bi))
        model.add(Dropout(0.2))
        model.add(Dense(400, activation='relu', kernel_initializer=ki, bias_initializer=bi))
        model.add(Dropout(0.3))
        model.add(Dense(200, activation='relu', kernel_initializer=ki, bias_initializer=bi))
        model.add(Dropout(0.05))
        model.add(Dense(units=self.action_size, activation='linear', kernel_initializer=ki, bias_initializer=bi))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy']
                      )
        return model

    #
    # Simple NN model learning QValues by state.
    #
    def _build_critic_model(self):
        ki = RandomUniform(minval=-0.05, maxval=0.05, seed=self.__seed)
        bi = Zeros()
        model = Sequential()
        model.add(Dense(800, input_dim=self.state_size, activation='relu', kernel_initializer=ki, bias_initializer=bi))
        model.add(Dropout(0.1))
        model.add(Dense(800, activation='relu', kernel_initializer=ki, bias_initializer=bi))
        model.add(Dropout(0.2))
        model.add(Dense(400, activation='relu', kernel_initializer=ki, bias_initializer=bi))
        model.add(Dropout(0.3))
        model.add(Dense(200, activation='relu', kernel_initializer=ki, bias_initializer=bi))
        model.add(Dropout(0.05))
        model.add(Dense(units=self.action_size, activation='linear', kernel_initializer=ki, bias_initializer=bi))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy']
                      )
        return model

    def critic_pred(self,
                    state: Tuple[int, int]) -> np.ndarray:
        """
        Return the critic (Value) prediction for the given state
        :param state: The current state as x, y position in state space
        :return: The q-value prediction of the critic network
        """
        st = np.array([state[0], state[1]])
        st = st.reshape([1, 2])
        return self.critic_model.predict(st, batch_size=1).flatten()

    def actor_pred(self,
                   state: Tuple[int, int]) -> np:
        return

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
                            np.round(next_state, self.state_dp)])
        if self.kl_update % 250 == 0:
            self.replay_kl_factor = self.replay_kl()
            self.kl_update = 0
        self.kl_update += 1
        return

    #
    # Act according to the current stochastic policy
    #
    def act(self,
            state) -> Tuple[int, float]:
        # state = state.reshape([1, state.shape[0]])
        klf = self.replay_kl_factor
        aprob = self.actor_model.predict(state, batch_size=1).flatten()
        aprob[aprob < 0.0] = 0.0
        if np.sum(aprob) == 0:
            aprob = np.array([.25, .25, .25, .25])
        else:
            aprob = ((1 - klf) * aprob) + (klf * np.array([.25, .25, .25, .25]))
            aprob /= np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=aprob)[0]
        return action, klf

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
                     episode: int) -> Tuple[float, float]:
        batch_size = min(len(self.replay), 250)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        samples = random.sample(list(self.replay), batch_size)
        i = 0
        for sample in samples:
            state, action_one_hot, reward, next_state = sample
            lr = self.qval_lr(episode)
            action_value_s = self.critic_model.predict(state, batch_size=1).flatten()
            action_value_ns = self.critic_model.predict(next_state, batch_size=1).flatten()
            qv_s = action_one_hot * action_value_s
            qv_ns = np.max(action_value_ns)  # * self.gamma
            av = (reward + (qv_ns * self.gamma)) - np.max(action_value_s)
            # av = ((qv_s * (1 - lr)) + (lr * (reward + qv_ns))) - np.max(action_value_s)
            qv_u = (action_value_s * (1 - action_one_hot)) + (av * action_one_hot)
            X[i] = np.squeeze(state[0])
            Y[i] = np.squeeze(qv_u)
            i += 1
        ls, acc = self.critic_model.train_on_batch(X, Y)
        print("Critic Training: episode [{:d}] - [{:f} - {:f}]".format(episode, ls, acc))
        return ls, acc

    #
    # Train the actor to learn the stochastic policy; the reward is the reward for the action
    # as predicted by the critic.
    #
    def train_actor(self,
                    lr: float) -> Tuple[float, float]:
        """
        ToDo: Leanring Rate Decay by Episode ?
        :return:
        """
        batch_size = min(len(self.replay), 250)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        samples = random.sample(list(self.replay), batch_size)
        i = 0
        for sample in samples:
            state, action_one_hot, reward, next_state = sample
            action_value_s = self.critic_model.predict(state, batch_size=1).flatten()
            action_probs_s = self.actor_model.predict(state, batch_size=1).flatten()

            avn = ((1 - action_one_hot) * action_value_s) + (action_one_hot * reward)
            avn -= np.max(avn)
            avn /= np.abs(np.sum(avn))

            action_probs_s[action_probs_s <= 0.0] = 0.01  # min % chance = 1%
            action_probs_s /= np.sum(action_probs_s)
            action_probs_s += (action_probs_s * avn * 0.7)
            action_probs_s /= np.sum(action_probs_s)

            X[i] = state
            Y[i] = action_probs_s
            i += 1
        ls, acc = self.actor_model.train_on_batch(X, Y)
        print("Actor Training: loss [{:f}] accuracy [{:f}]".format(ls, acc))
        return ls, acc

    #
    # Return the state space and predicted probabilities for given states
    #
    def probs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mn = self.env.state_min()
        mx = self.env.state_max()
        st = self.env.state_step()
        x = np.arange(mn, mx, st)
        y = np.arange(mn, mx, st)
        z = np.zeros((x.shape[0], y.shape[0], self.env.num_actions()))
        i = 0
        j = 0
        for sx in x:
            for sy in y:
                state = self.env.state_as_x((sx, sy))
                z[i, j] = self.actor_model.predict(state, batch_size=1).flatten()
                j += 1
            i += 1
            j = 0
        return x, y, z

    #
    # Return the state space and predicted critic reward values
    #
    def vals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mn = self.env.state_min()
        mx = self.env.state_max()
        st = self.env.state_step()
        x = np.arange(mn, mx, st)
        y = np.arange(mn, mx, st)
        z = np.zeros((x.shape[0], y.shape[0], self.env.num_actions()))
        i = 0
        j = 0
        for sx in x:
            for sy in y:
                state = self.env.state_as_x((sx, sy))
                z[i, j] = self.critic_model.predict(state, batch_size=1).flatten()
                #  z[i, j] = np.array([
                #      self.env.reward((sx, sy+1)),  # N
                #      self.env.reward((sx+1, sy)),  # E
                #      self.env.reward((sx, sy-1)),  # S
                #      self.env.reward((sx-1, sy))   # W
                #  ])
                j += 1
            i += 1
            j = 0
        return x, y, z

    #
    # Simple debugger output - could be refactored into PBFunc env as it is more env specific ?
    #
    def print_progress(self,
                       ep: int,
                       elen: int) -> None:
        print(str(ep) + "::" + str(elen))
        self.actor_loss_history = self.actor_loss_history[-500:]
        self.critic_loss_history = self.critic_loss_history[-500:]
        self.actor_acc_history = self.actor_acc_history[-500:]
        self.critic_acc_history = self.critic_acc_history[-500:]
        self.actor_exploration_history = self.actor_exploration_history[-500:]

        self.visualise().plot_loss_function(actor_loss=self.actor_loss_history,
                                            critic_loss=self.critic_loss_history)

        # self.visualise().plot_acc_function(actor_acc=self.actor_acc_history,
        #                                   critic_acc=self.critic_acc_history,
        #                                   exploration=self.actor_exploration_history)

        # self.visualise().plot_qvals_function(states=states,
        #                                     qvalues_action1=predicted_qval_action1,
        #                                     qvalues_action2=predicted_qval_action2,
        #                                     qvalues_reference=replay_qvals)

        self.visualise().plot_prob_function(self.probs)
        self.visualise().plot_qvals_function(self.vals)
        return


#
# Main Function.
#
class Main:

    @classmethod
    def run(cls,
            reward_function_2d: RewardFunction2D):
        env = reward_function_2d

        st = env.reset()
        episode = 0
        eln = 0

        state_size = env.state_space_dimension()
        action_size = env.num_actions()
        agent = PolicyGradientAgent2D(reward_function_2d=env,
                                      st_size=state_size,
                                      a_size=action_size,
                                      num_states=(env.state_max() - env.state_min()) / env.state_step())

        agent.visualise().plot_reward_function(func=env.func)

        als = None
        rls = None
        elau = 0
        thr = 50
        accc = 0
        acca = 0
        while True:
            a, expl = agent.act(st)
            next_state, reward, done = env.step(a)
            agent.remember(st, a, reward, next_state)
            st = next_state
            eln += 1

            acl = 0.1
            lr0 = 0.1
            if done or eln > 250:
                if episode > 3:
                    rls, accc = agent.train_critic(episode)
                    acl = accc
                if episode > 3:
                    thr = (100 - (np.round(accc * 100, 0)))
                    if thr <= 15:
                        thr = 2
                    if (episode - elau) > thr:
                        print(max(5, (100 - (np.round(accc * 10, 0) * 10))))
                        print('<<<********** Train actor *************>>>')
                        als, acca = agent.train_actor(lr0 * acl)  # * aal)
                        elau = episode
                if episode > 1 and episode % 100 == 0:
                    agent.critic_loss_history.append(rls)
                    agent.actor_loss_history.append(als)
                    agent.actor_acc_history.append(acca)
                    agent.critic_acc_history.append(accc)
                    agent.actor_exploration_history.append(expl)
                    agent.print_progress(episode, eln)
                print("Episode - Episode-Length: {e}-{el}".format(e=episode, el=eln))
                episode += 1
                eln = 0
                st = env.reset()


#
# Endless loop of agent acting / learning in the given env.
#
if __name__ == "__main__":
    local_maxima_reward = LocalMaximaRewardFunction2D()
    Main.run(local_maxima_reward)
