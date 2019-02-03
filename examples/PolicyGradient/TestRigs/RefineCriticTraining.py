import random
from collections import deque

import numpy as np
from keras.initializers import RandomUniform
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
        self.critic_model = self._build_critic_model()

        self.critic_model.summary()
        qval_lr0 = float(1)
        self.qval_learning_rate = SimpleLearningRate(lr0=qval_lr0,
                                                     lrd=SimpleLearningRate.lr_decay_target(learning_rate_zero=qval_lr0,
                                                                                            target_step=5000,
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
    # Simple NN model learning QValues by state.
    #
    def _build_critic_model(self):
        ru = RandomUniform(minval=-0.05, maxval=0.05, seed=None)

        model = Sequential()
        model.add(Dense(1000, input_dim=self.state_size, activation='relu', kernel_initializer=ru))
        model.add(Dense(500, activation='relu', kernel_initializer=ru))
        model.add(Dense(100, activation='relu', kernel_initializer=ru))
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
    def act(self) -> int:
        aprob = np.array([.5, .5])
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
        batch_size = min(len(self.replay), 250)
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
            qv_ns = np.max(action_value_ns)  # * self.gamma
            av = (reward + (qv_ns * self.gamma)) - np.max(action_value_s)
            # av = ((qv_s * (1 - lr)) + (lr * (reward + qv_ns))) - np.max(action_value_s)
            qv_u = (action_value_s * (1 - action_one_hot)) + (av * action_one_hot)
            X[i] = np.squeeze(state)
            Y[i] = np.squeeze(qv_u)
            i += 1
        ls, acc = self.critic_model.train_on_batch(X, Y)
        print("Critic Training: episode [{:d}] loss [{:f}] accuracy [{:f}]".format(episode, ls, acc))
        return ls

    #
    # Simple debugger output - could be refactored into PBFunc env as it is more env specific ?
    #
    def print_progress(self,
                       ep: int,
                       elen: int,
                       e: RewardFunction1D) -> None:
        res = str()
        ts = e.state_min()
        predicted_qval_action1 = []
        predicted_qval_action2 = []
        states = []
        replay_qvals = []
        for sv in np.arange(e.state_min(), e.state_max(), e.state_step()):
            state = e.state_as_x(sv)
            states.append(state[0])
            replay_qvals.append(e.reward(sv))
            qvals = self.critic_model.predict(state, batch_size=1).flatten()
            predicted_qval_action1.append(qvals[0])
            predicted_qval_action2.append(qvals[1])
            # print("S: " + '{:+.2}'.format(float(ts)) + " [ " +
            #      str(round(0 * 100, 2)) + "% , " +
            #     str(round(0 * 100, 2)) + "%], {" +
            #      str(round(qvals[0], 4)) + "} {" +
            #      str(round(qvals[1], 4)) + "}"
            #      )
        self.visualise().plot_loss_function(actor_loss=self.actor_loss_history,
                                            critic_loss=self.critic_loss_history)
        self.visualise().plot_qvals_function(states=states,
                                             qvalues_action1=predicted_qval_action1,
                                             qvalues_action2=predicted_qval_action2,
                                             qvalues_reference=replay_qvals)
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
            a = agent.act()
            next_state, reward, done = env.step(a)
            agent.remember(st, a, reward, next_state)
            st = next_state
            eln += 1

            if done or eln > 500:
                if episode > 3:
                    rls = agent.train_critic(episode)
                if episode > 1 and episode % 3 == 0:
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
if __name__ == "__main__":
    parabolic_reward = ParabolicRewardFunction1D()
    local_maxima_reward = LocalMaximaRewardFunction1D()
    Main.run(local_maxima_reward)
