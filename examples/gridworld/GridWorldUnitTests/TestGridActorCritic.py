import random
from collections import deque

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.models import Model
from keras.optimizers import Adam

from examples.gridworld.SimpleGridOne import SimpleGridOne


#
# Learn values for each state given state and action.
#

class TestGridActorCritic:
    def __init__(self, grid, sess):
        self.env_grid = grid
        self.env_action_space_shape = [1]  # 4 Actions, but shape is [1] as it is not one-hot encoded.
        self.env_observation_space_shape = [2]
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95  # Discount Factor Applied to reward
        self.tau = .125

        #
        # This is the replay-memory, this is needed so the target is "stationary" and the
        # model converges. This is so as the reply memory is randomly sampled periodically
        # in batches and then used for supervised learning on the "latest" set of weights.
        #
        self.memory = deque(maxlen=2000)

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A select_action #
        # ===================================================================== #

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env_action_space_shape[
                                                    0]])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calculate de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    #
    # Actor : Given state, predict action.
    # Critic: Given state & action
    #
    # Must have same architecture as critic gradients used to train actor.
    #
    def create_actor_model(self):
        state_input = Input(shape=self.env_observation_space_shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env_action_space_shape[0], activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env_observation_space_shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env_action_space_shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], np.array(reward).reshape(1, 1), verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    #
    # Predict an allowable action.
    #
    # ToDo: possibility that this will not exit, need throw exception if after n tries actor has not picked
    # ToDo: an allowable action, or treat this as a case to return random action (exploration)
    #
    def actor_predict_action(self, cur_state):
        allowable_actions = self.env_grid.allowable_actions(cur_state)
        cur_state = np.array(cur_state).reshape((1, env_observation_space_shape[0]))  # Shape needed for NN
        actn = int(np.round(self.actor_model.predict(cur_state)))  # convert from float to action (int)
        mx = 0
        while actn not in allowable_actions and mx < 10:
            actn = int(np.round(self.actor_model.predict(cur_state)))  # convert from float to action (int)
            mx += 1
        if mx == 10:
            actn = None
        return actn

    #
    # Make a greedy (random) action based on current value (decaying) of epsilon
    # else make an action based on prediction the NN inside the actor.
    #
    def select_action(self, cur_state):
        if self.env_grid.episode_complete():  # At goal state, so re spawn to start point.
            self.env_grid.reset()
            cur_state = self.env_grid.state()
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            actns = self.env_grid.allowable_actions(cur_state)
            if len(actns) > 0:
                return random.choice(actns)
            else:
                print("??")
        actn = self.actor_predict_action(cur_state)
        if actn is None:
            return random.choice(self.env_grid.allowable_actions(cur_state))
        else:
            return actn


#
# Return a 5 by 5 grid as the environment. The aim is to learn the shortest
# path from the bottom right to the goal at (1,1), while avoiding the
# penalty at (4,4). Each step has a negative cost, and it is this that drives
# the push for shortest path in terms of minimising the cost function.
#
# The action space is N,S,E,W shape (1,4)
# The state is the grid coordinates (x,y) shape (1,2)
#
step = SimpleGridOne.STEP
fire = SimpleGridOne.FIRE
blck = SimpleGridOne.BLCK
goal = SimpleGridOne.GOAL


def create_grid() -> SimpleGridOne:
    grid = [
        [step, step, step, step, step],
        [step, goal, step, step, step],
        [step, step, step, step, step],
        [step, step, step, fire, step],
        [step, step, step, step, step]
    ]
    sg1 = SimpleGridOne(1,
                        grid,
                        [4, 4])
    return sg1


env_observation_space_shape = [2]
env_action_space_shape = [1]


def main():
    sess = tf.Session()
    K.set_session(sess)
    env = create_grid()
    actor_critic = TestGridActorCritic(env, sess)

    env.reset()
    cur_state = env.state()

    episode = 0
    while True:
        action = actor_critic.select_action(cur_state)
        reward = env.execute_action(action)
        new_state = env.state()
        done = env.episode_complete()

        cur_state = np.array(cur_state).reshape((1, env_observation_space_shape[0]))
        new_state = np.array(new_state).reshape((1, env_observation_space_shape[0]))
        action = np.array(action).reshape((1, env_action_space_shape[0]))
        actor_critic.remember(cur_state, action, reward, new_state, done)

        actor_critic.train()

        cur_state = env.state()  # new_state

        episode += 1
        print(episode)
        if episode % 100 == 0:
            for i in range(0, 4):
                s = ""
                for j in range(0, 4):
                    st = np.array([i, j]).reshape((1, env_observation_space_shape[0]))
                    s += str(actor_critic.critic_model.predict(st)[0])
                    s += ' , '
                print(s)


if __name__ == "__main__":
    main()
