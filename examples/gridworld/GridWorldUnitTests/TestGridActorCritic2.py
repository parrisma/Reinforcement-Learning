import logging
import random
from collections import deque
from random import shuffle

import keras.backend as K
import numpy as np
import tensorflow as tf

from examples.gridworld.RenderSimpleGridOneQValues import RenderSimpleGridOneQValues
from examples.gridworld.SimpleGridOne import SimpleGridOne
from examples.gridworld.TestRigs.GridWorldQValNNModel import GridWorldQValNNModel
from reflrn.EnvironmentLogging import EnvironmentLogging


#
# Learn values for each state given state and action.
#

class TestGridActorCritic:
    def __init__(self,
                 grid,
                 sess,
                 lg,
                 rows: int,
                 cols: int):
        self.env_grid = grid
        self.sess = sess
        self.lg = lg

        self.epoch = 0

        self.batch_size = 32
        self.input_dim = 2
        self.output_dim = self.num_actions
        self.num_actions = 4
        self.num_rows = rows
        self.num_cols = cols

        self.learning_rate_0 = float(1.0)
        self.learning_rate_decay = float(0.05)
        self.epsilon = 0.8  # exploration factor.
        self.epsilon_decay = .9995
        self.gamma = .95  # Discount Factor Applied to reward

        self.exploration = dict()

        #
        # This is the replay-memory, this is needed so the target is "stationary" and the
        # model converges. This is so as the reply memory is randomly sampled periodically
        # in batches and then used for supervised learning on the "latest" set of weights.
        #
        self.memory = deque(maxlen=2000)
        self.goal_memory = deque(maxlen=50)

        self.actor_model = GridWorldQValNNModel(model_name="Actor",
                                                input_dimension=self.input_dim,
                                                num_actions=self.num_actions,
                                                num_grid_cells=(self.num_rows * self.num_cols),
                                                lg=self.lg,
                                                batch_size=self.batch_size,
                                                num_epoch=2,
                                                lr_0=0.005,
                                                lr_min=0.001
                                                )

        self.critic_model = GridWorldQValNNModel(model_name="Critic",
                                                 input_dimension=self.input_dim,
                                                 num_actions=self.num_actions,
                                                 num_grid_cells=(self.num_rows * self.num_cols),
                                                 lg=self.lg,
                                                 batch_size=self.batch_size,
                                                 num_epoch=2,
                                                 lr_0=0.005,
                                                 lr_min=0.001
                                                 )
        return

    #
    # Return the learning rate based on number of learning's to date
    #
    def learning_rate(self, ):
        return self.learning_rate_0 / (1 + (self.epoch * self.learning_rate_decay))

    #
    # Add item to the replay-memory.
    #
    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])
        if reward == SimpleGridOne.GOAL:
            self.goal_memory.append([cur_state, action, reward, new_state, done])

    #
    # Actor is not trained, but instead clones the trainable parameters from the critic
    # after every n times the critic is trained on a replay memory batch.
    #
    def _update_actor_from_critic(self):
        self.actor_model.clone_weights(self.critic_model)

    def _train_critic(self) -> bool:
        trained = False
        x, y = self._get_sample_batch()
        if x is not None:
            self.critic_model.train(x, y)
            trained = True
        return trained

    #
    # Get a random set of samples from the given QValues to select_action as a training
    # batch for the model.
    #
    def _get_sample_batch(self):
        if len(self.memory) < self.batch_size:
            return None, None

        x = np.zeros((self.batch_size, 2))
        y = np.zeros((self.batch_size, self.num_actions))
        actn = np.arange(0, self.num_actions, 1)

        if len(self.goal_memory) > 0:
            samples_g = random.sample(self.goal_memory, 1)
            samples = random.sample(self.memory, self.batch_size - 1)
            for s in samples_g:
                samples.append(s)
            shuffle(samples)
        else:
            samples = random.sample(self.memory, self.batch_size)
        i = 0
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if reward == SimpleGridOne.GOAL:
                print("Goal Reward")
            x[i] = cur_state
            qvs = self._actor_prediction(cur_state)[0]  # Actor estimate of QVals for current state.
            qvn = self._actor_prediction(new_state)[0]  # Actor estimate of QVals for next state. (after action)
            lr = self.learning_rate()

            qvp = np.max(qvn)
            qvp = self.gamma * qvp * lr  # Discounted max return from next state

            qv = qvs[action]
            qv = (qv * (1 - lr)) + (lr * reward) + qvp  # updated expectation of current state/action
            y[i][action] = qv
            i += 1
        return x, y

    #
    # Train after every 10 epoch (goal states reached)
    #
    def train(self):
        if self._train_critic():
            if self.epoch % 10 == 0:
                self._update_actor_from_critic()
        return

    #
    # Get actor prediction, if actor is not able to predict, predict random
    #
    def _actor_prediction(self, cur_state):
        z = np.zeros((1, 2))
        z[0] = cur_state
        p = self.actor_model.predict(z)
        if p is None:
            p = 0.5 * np.random.random_sample(self.num_actions) - 0.5
        return p

    #
    # Predict an allowable action.
    #
    def actor_predict_action(self, cur_state, allowable_actions):
        actn = self._get_actor_predicted_action(cur_state)
        mx = 0
        while actn not in allowable_actions and mx < 10:
            actn = self._get_actor_predicted_action(cur_state)
            mx += 1
        if mx == 10:
            actn = None
        return actn

    #
    # Get the action predicted by the actor for the given state
    #
    def _get_actor_predicted_action(self, cur_state) -> int:
        cur_state = np.array(cur_state).reshape((1, 2))  # Shape needed for NN
        actn = np.argmax(self.actor_model.predict(cur_state))
        return actn

    #
    # Select a random action, but bias to taking the action that has been least
    # explored from this state.
    #
    def remember_action(self, cur_state, action):
        ky = str(cur_state)
        if ky not in self.exploration:
            self.exploration[ky] = np.zeros(self.num_actions)
        self.exploration[ky][action] += 1
        return

    #
    # Select a random action, but prefer an action not or least taken from
    # the current state (this should be replaced with another NN that learns
    # familiarity)
    #
    def random_action(self, cur_state, allowable_actions):
        ky = str(cur_state)
        if ky in self.exploration:
            act_cnt = self.exploration[ky]
            actn = allowable_actions[np.argmin(act_cnt[allowable_actions])]
        else:
            actn = random.choice(allowable_actions)
        return actn

    #
    # Make a greedy (random) action based on current value (decaying) of epsilon
    # else make an action based on prediction the NN inside the actor.
    #
    def select_action(self, cur_state):
        if self.env_grid.episode_complete():  # At goal state, so re spawn to start point.
            self.lg.debug("Goal Reached")
            self.env_grid.reset()
            cur_state = self.env_grid.state()
            self.epoch += 1

        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)
        self.lg.debug("epsilon :" + str(self.epsilon))
        allowable_actions = self.env_grid.allowable_actions(cur_state)

        if np.random.random() < self.epsilon:
            self.lg.debug("R")
            actn = self.random_action(cur_state, allowable_actions)
        else:
            actn = self.actor_predict_action(cur_state, allowable_actions)
            if actn is None:
                self.lg.debug("R")
                actn = self.random_action(cur_state, allowable_actions)
            else:
                self.lg.debug("P")

        self.remember_action(cur_state, actn)
        return actn


#
# Return a 5 by 5 grid as the environment. The aim isto learn the shortest
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
        [step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, goal],
        [step, step, step, step, step, step, step, step, step, step],
        # [step, step, step, step, step, step, step, step, step, step],
        # [step, step, step, step, step, step, step, step, step, step],
        # [step, step, step, step, step, step, step, step, step, step],
        # [step, step, step, step, step, step, step, step, step, step],
        # [step, step, step, step, step, step, step, fire, step, step],
        # [step, step, step, step, step, step, step, step, step, step],
        # [step, step, step, step, step, step, step, step, step, step],
    ]
    sg1 = SimpleGridOne(1,
                        grid,
                        [0, 0])
    return 3, 10, sg1


#
#
#
def main():
    lg = EnvironmentLogging("TestGridActorCritic2",
                            "TestGridActorCritic2.log",
                            logging.DEBUG).get_logger()

    sess = tf.Session()
    K.set_session(sess)
    r, c, env = create_grid()
    actor_critic = TestGridActorCritic(env, sess, lg, r, c)
    rdr = RenderSimpleGridOneQValues(num_cols=actor_critic.num_cols,
                                     num_rows=actor_critic.num_rows,
                                     plot_style=RenderSimpleGridOneQValues.PLOT_SURFACE,
                                     do_plot=True)

    env.reset()
    cur_state = env.state()

    episode = 0
    while True:
        action = actor_critic.select_action(cur_state)

        reward = env.execute_action(action)
        new_state = env.state()
        done = env.episode_complete()
        actor_critic.remember(cur_state, action, reward, new_state, done)
        lg.debug(":: " + str(cur_state) + " -> " + str(action) + " = " + str(reward))

        actor_critic.train()

        cur_state = env.state()  # new_state

        # Visualize.

        episode += 1
        print(episode)
        qgrid = np.zeros((actor_critic.num_rows, actor_critic.num_cols))
        if episode % 20 == 0:
            for i in range(0, actor_critic.num_rows):
                s = ""
                for j in range(0, actor_critic.num_cols):
                    st = np.array([i, j]).reshape((1, actor_critic.input_dim))
                    q_vals = actor_critic.critic_model.predict(st)[0]
                    s += str(['N', 'S', 'E', 'W'][np.argmax(q_vals)])
                    s += ' , '

                    for actn in range(0, actor_critic.num_actions):
                        x, y = SimpleGridOne.coords_after_action(i, j, actn)
                        if x >= 0 and y >= 0 and x < actor_critic.num_rows and y < actor_critic.num_cols:
                            if qgrid[x][y] == np.float(0):
                                qgrid[x][y] = q_vals[actn]
                            else:
                                qgrid[x][y] += q_vals[actn]
                                qgrid[x][y] /= np.float(2)
                print(s)
                lg.debug(s)
            rdr.plot(qgrid)
            lg.debug('--------------------')


if __name__ == "__main__":
    main()
