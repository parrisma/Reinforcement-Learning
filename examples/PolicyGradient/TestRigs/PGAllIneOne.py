import numpy as np
from typing import Tuple
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class PbFunc:
    state_min = -1
    state_max = 1
    state_step = 0.05

    @classmethod
    def float_as_x(cls,
                   x: float):
        xs = np.array([x])
        return xs.reshape([1, xs.shape[0]])

    def reset(self) -> np.array:
        if np.random.rand() >= 0.5:
            self.state = self.state_max
        else:
            self.state = self.state_min
        return np.array([self.state])

    def __init__(self):
        self.state = None

        self.reset()

    @classmethod
    def state_space_size(cls) -> int:
        return 1

    @classmethod
    def num_actions(cls) -> int:
        return 2

    @classmethod
    def reward(cls,
               st: float) -> float:
        if -0.05 <= st <= 0.05:
            return float(0)
        return -(st * st)

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


class PGAgent:
    def __init__(self, st_size, a_size):
        self.state_size = st_size
        self.action_size = a_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.labels = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, r, pb):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.labels.append(np.array(y).astype('float32'))
        self.states.append(state)
        self.rewards.append(r)
        self.probs.append(pb)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        action = np.random.choice(self.action_size, 1, p=aprob)[0]
        return action, aprob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        labels = np.vstack(self.labels)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        # rewards = rewards / np.std(rewards)
        labels *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = np.squeeze(np.vstack([labels]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.labels, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def print_progress(self,
                       ep: int,
                       elen: int,
                       e: PbFunc) -> None:
        res = str()
        ts = e.state_min
        while ts <= e.state_max:
            ss = PbFunc.float_as_x(ts)
            aprob = self.model.predict(ss, batch_size=1).flatten()
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


if __name__ == "__main__":
    env = PbFunc()
    st = env.reset()
    score = 0
    episode = 0
    eln = 0

    state_size = env.state_space_size()
    action_size = env.num_actions()
    agent = PGAgent(state_size, action_size)
    while True:
        a, pbs = agent.act(st)
        next_state, reward, done = env.step(a)
        score += reward
        agent.remember(st, a, reward, pbs)
        st = next_state
        eln += 1

        if done or eln > 10000:
            if np.size(agent.rewards) > 10:
                episode += 1
                agent.rewards[-1] = score
                agent.train()
                agent.print_progress(episode, eln, env)
                score = 0
                eln = 0
                st = env.reset()
                if episode > 1 and episode % 50 == 0:
                    pass
