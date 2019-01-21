import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from examples.PolicyGradient.TestRigs.Interface.RewardFunction1D import RewardFunction1D
from examples.PolicyGradient.TestRigs.RewardFunctions.LocalMaximaRewardFunction1D import LocalMaximaRewardFunction1D
from examples.PolicyGradient.TestRigs.RewardFunctions.ParabolicRewardFunction1D import ParabolicRewardFunction1D
from examples.PolicyGradient.TestRigs.Visualise import Visualise
from reflrn.SimpleLearningRate import SimpleLearningRate


#
# Two network actor / critic stochastic policy with the critic learning state q-values on a one step Bellman.
#
# Exploration is inherent as policy is stochastic
#
class PolicyGradientAgent:
    __fig1 = None
    __fig2 = None
    __plot_pause = 0.0001

    def __init__(self,
                 st_size,
                 a_size):
        self.state_size = st_size
        self.action_size = a_size
        self.gamma = 0.99
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
                                                     lr_min=0.001)

        self.state_dp = 5
        self.critic_loss_history = []
        self.actor_loss_history = []

        self.visual = Visualise()
        self.visual.show()

        return

    def visualise(self) -> Visualise:
        return self.visual

    #
    # Simple NN model with softmax learning the policy as probability distribution over actions.
    #
    def _build_actor_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy']
                      )
        return model

    #
    # Simple NN model learning QValues by state.
    #
    def _build_critic_model(self):
        model = Sequential()
        model.add(Dense(2000, input_dim=self.state_size, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(1000, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(500, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(250, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(100, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(units=self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy']
                      )
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
                            np.round(next_state, self.state_dp)])
        return

    #
    # Act according to the current stochastic policy
    #
    def act(self,
            state,
            episode: int) -> int:
        state = state.reshape([1, state.shape[0]])
        lr = self.qval_lr(episode)
        aprob = self.actor_model.predict(state, batch_size=1).flatten()
        aprob = np.array([aprob[0], aprob[1]])
        aprob = (np.array([.5, .5]) * lr) + (aprob * (1.0 - lr))
        aprob /= np.sum(aprob)
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
                     episode: int) -> float:
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
            av = reward
            # av = (qv_s * (1 - lr)) + (lr * (reward + qv_ns))  # updated expectation of current state/action
            qv_u = (action_value_s * (1 - action_one_hot)) + (av * action_one_hot)
            X[i] = np.squeeze(state)
            Y[i] = np.squeeze(qv_u)
            i += 1
        ls, acc = self.critic_model.train_on_batch(X, Y)
        print("Critic Training: episode [{:d}] loss [{:f}] accuracy [{:f}]".format(episode, ls, acc))
        return ls

    #
    # Train the actor to learn the stochastic policy; the reward is the reward for the action
    # as predicted by the critic.
    #
    def train_actor(self) -> float:
        """
        ToDo: Leanring Rate Decay by Episode ?
        :return:
        """
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
        ls, acc = self.actor_model.train_on_batch(X, Y)
        print("Actor Training: loss [{:f}] accuracy [{:f}]".format(ls, acc))
        return ls

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
                       e: RewardFunction1D) -> None:
        res = str()
        ts = e.state_min()
        predicted_prob_action1 = []
        predicted_prob_action2 = []
        predicted_qval_action1 = []
        predicted_qval_action2 = []
        states = []
        replay_qvals = []
        for sv in np.arange(e.state_min(), e.state_max(), e.state_step()):
            state = e.state_as_x(sv)
            states.append(state[0])
            replay_qvals.append(e.reward(sv))
            aprob = self.actor_model.predict(state, batch_size=1).flatten()
            qvals = self.critic_model.predict(state, batch_size=1).flatten()
            predicted_prob_action1.append(aprob[0])
            predicted_prob_action2.append(aprob[1])
            predicted_qval_action1.append(qvals[0])
            predicted_qval_action2.append(qvals[1])
            print("S: " + '{:+.2}'.format(float(ts)) + " [ " +
                  str(round(aprob[0] * 100, 2)) + "% , " +
                  str(round(aprob[1] * 100, 2)) + "%], {" +
                  str(round(qvals[0], 4)) + "} {" +
                  str(round(qvals[1], 4)) + "}"
                  )
            if aprob[0] > aprob[1]:
                res += '>'
            else:
                res += "<"
            if ts == 0:
                res += "|"
            ts += e.state_step()
        print(str(ep) + "::" + str(elen) + "   " + res)
        self.visualise().plot_loss_function(actor_loss=self.actor_loss_history,
                                            critic_loss=self.critic_loss_history)
        self.visualise().plot_qvals_function(states=states,
                                             qvalues_action1=predicted_qval_action1,
                                             qvalues_action2=predicted_qval_action2,
                                             qvalues_reference=replay_qvals)
        self.visualise().plot_prob_function(states=states,
                                            action1_probs=predicted_prob_action1,
                                            action2_probs=predicted_prob_action2)
        return


class Test:
    @classmethod
    def run(cls,
            reward_function_1d: RewardFunction1D):
        v = Visualise()
        v.show()
        env = reward_function_1d
        states, rewards = env.func()
        v.plot_reward_function(states=states, rewards=rewards)
        state_size = env.state_space_size()
        action_size = env.num_actions()
        agent = PolicyGradientAgent(state_size, action_size)
        PolicyGradientAgent.gamma = 0
        for i in range(0, 15):
            j = float(reward_function_1d.state_min())
            while j <= reward_function_1d.state_max():
                cs = reward_function_1d.state_as_x(j)
                nsa0 = reward_function_1d.state_as_x(j + reward_function_1d.state_step())
                nsa1 = reward_function_1d.state_as_x(j - reward_function_1d.state_step())
                agent.remember(cs, 0, reward_function_1d.reward(nsa0), nsa0)
                agent.remember(cs, 1, reward_function_1d.reward(nsa1), nsa1)
                j += reward_function_1d.state_step()
        ls = []
        for i in range(0, 1000):
            ls.append(agent.train_critic(i))
            if i % 10 == 0:
                samples = random.sample(list(agent.replay), 200)
                predicted_qvals_action1 = []
                predicted_qvals_action2 = []
                states = []
                replay_qvals = []
                for sample in samples:
                    state, action_one_hot, reward, next_state = sample
                    states.append((state[0])[0])
                    replay_qvals.append(reward)
                    qvs = agent.critic_model.predict(state, batch_size=1).flatten()
                    predicted_qvals_action1.append(qvs[0])
                    predicted_qvals_action2.append(qvs[1])
                v.plot_qvals_function(states,
                                      predicted_qvals_action1,
                                      predicted_qvals_action2,
                                      replay_qvals)
                v.plot_loss_function(actor_loss=None, critic_loss=ls)
        return


#
# Test Rig Main Function.
#
class Main:

    @classmethod
    def run(cls,
            reward_function_1d: RewardFunction1D):
        env = reward_function_1d

        st = env.reset()
        episode = 0
        eln = 0

        state_size = env.state_space_size()
        action_size = env.num_actions()
        agent = PolicyGradientAgent(state_size, action_size)

        states, rewards = env.func()
        agent.visualise().plot_reward_function(states=states, rewards=rewards)

        als = None
        rls = None
        while True:
            a = agent.act(st, episode)
            next_state, reward, done = env.step(a)
            agent.remember(st, a, reward, next_state)
            st = next_state
            eln += 1

            if done or eln > 500:
                if episode > 3:
                    rls = agent.train_critic(episode)
                if episode > 1 and episode % 3 == 0:
                    als = agent.train_actor()
                    agent.critic_loss_history.append(rls)
                    agent.actor_loss_history.append(als)
                    agent.print_progress(episode, eln, env)
                print("Episode - Episode-Length: {e}-{el}".format(e=episode, el=eln))
                episode += 1
                eln = 0
                st = env.reset()


#
# Endless loop of agent acting / learning in the given env.
#
test = False
if __name__ == "__main__":
    parabolic_reward = ParabolicRewardFunction1D()
    local_maxima_reward = LocalMaximaRewardFunction1D()
    if test:
        Test.run(parabolic_reward)
    else:
        Main.run(local_maxima_reward)
