import logging
import random

import keras.backend as K
import numpy as np
import tensorflow as tf

from examples.gridworld.RenderSimpleGridOneQValues import RenderSimpleGridOneQValues
from examples.gridworld.SimpleGridOne import SimpleGridOne
from examples.gridworld.TestRigs.GridWorldQValNNModel import GridWorldQValNNModel
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.RareEventBiasReplayMemory import RareEventBiasReplayMemory


#
# Actor critic pattern to learn to find goals on a simple  2D grid.
#

class GridActorCritic:
    def __init__(self,
                 grid,
                 sess,
                 lg,
                 rows: int,
                 cols: int):
        self.env_grid = grid
        self.sess = sess
        self.lg = lg

        self.episode = 1

        self.batch_size = 32
        self.input_dim = 2
        self.num_actions = 4
        self.output_dim = self.num_actions
        self.num_rows = rows
        self.num_cols = cols

        self.learning_rate_0 = float(1.0)
        self.learning_rate_decay = float(0.05)
        self.epsilon = 0.8  # exploration factor.
        self.epsilon_decay = .9995
        self.gamma = .8  # Discount Factor Applied to reward

        self.exploration = dict()

        self.steps_to_goal = 0
        self.train_on_new_episode = False

        #
        # This is the replay-memory, this is needed so the target is "stationary" and the
        # model converges. This is so as the reply memory is randomly sampled periodically
        # in batches and then used for supervised learning on the "latest" set of weights.
        #
        self.replay_memory = RareEventBiasReplayMemory(self.lg, replay_mem_size=2000)

        self.actor_model = GridWorldQValNNModel(model_name="Actor",
                                                input_dimension=self.input_dim,
                                                num_actions=self.num_actions,
                                                num_grid_cells=(self.num_rows * self.num_cols),
                                                lg=self.lg,
                                                batch_size=self.batch_size,
                                                num_epoch=1,
                                                lr_0=0.005,
                                                lr_min=0.001
                                                )

        self.critic_model = GridWorldQValNNModel(model_name="Critic",
                                                 input_dimension=self.input_dim,
                                                 num_actions=self.num_actions,
                                                 num_grid_cells=(self.num_rows * self.num_cols),
                                                 lg=self.lg,
                                                 batch_size=self.batch_size,
                                                 num_epoch=1,
                                                 lr_0=0.005,
                                                 lr_min=0.001
                                                 )
        return

    #
    # Return the learning rate based on number of learning's to date
    #
    def learning_rate(self):
        return max(0.01, self.learning_rate_0 / (1 + (self.episode * self.learning_rate_decay)))

    #
    # Add item to the replay-memory.
    #
    def remember(self, cur_state, action, reward, new_state, done):
        self.replay_memory.append_memory(state=cur_state,
                                         action=action,
                                         reward=reward,
                                         next_state=new_state,
                                         episode_complete=done)

    #
    # Actor is not trained, but instead clones the trainable parameters from the critic
    # after every n times the critic is trained on a replay memory batch.
    #
    def _update_actor_from_critic(self):
        self.actor_model.clone_weights(self.critic_model)
        self.lg.debug("Update Actor From Critic")
        return

    def _train_critic(self) -> bool:
        trained = False
        x, y = self._get_sample_batch()
        if x is not None:
            self.critic_model.train(x, y)
            trained = True
            self.lg.debug("Critic Trained")
        return trained

    #
    # Convert the allowable actions into a boolean mask.
    #
    def disallowed_actions(self, allowable_actions):
        da = []
        for i in range(0, self.num_actions):
            if i not in allowable_actions:
                da.append(i)
        return np.array(da)

    #
    # Get a random set of samples from the given QValues to select_action as a training
    # batch for the model.
    #
    def _get_sample_batch(self):
        try:
            samples = self.replay_memory.get_random_memories(self.batch_size)
        except RareEventBiasReplayMemory.SampleMemoryTooSmall:
            return None, None

        x = np.zeros((self.batch_size, self.input_dim))
        y = np.zeros((self.batch_size, self.num_actions))
        i = 0
        for sample in samples:
            cur_state, new_state, action, reward, done = sample
            aact = self.env_grid.allowable_actions(cur_state)
            da = self.disallowed_actions(aact)
            lr = self.learning_rate()

            qvn = self._actor_prediction(new_state)[0]  # Actor estimate of QVals for next state. (after action)
            qvp = np.max(qvn[aact])  # optimal return, can only be taken from allowable actions.
            qvp = self.gamma * qvp * lr  # Discounted max return from next state

            qvs = self._actor_prediction(cur_state)[0]  # Actor estimate of QVals for current state.
            if len(aact) < self.num_actions:
                lv = np.min(qvs[aact])
                lv = np.absolute(lv) * 1.01 * np.sign(lv)
                qvs[da] = lv  # suppress disallowed action by making qval less then smallest allowable qval / actn.
            qv = qvs[action]
            qv = (qv * (1 - lr)) + (lr * reward) + qvp  # updated expectation of current state/action
            qvs[action] = qv

            x[i] = cur_state
            y[i] = qvs
            i += 1
        return x, y

    #
    # Update actor every 5 episodes (goal states reached)
    #
    def train(self):
        if self.replay_memory.get_num_memories() > 100:  # don't start learning until we have reasonable # mems
            if self._train_critic():
                if self.train_on_new_episode:
                    self._update_actor_from_critic()
                    self.train_on_new_episode = False
                else:
                    if self.steps_to_goal % 100 == 0:
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
        actn = np.argmax(self.actor_model.predict(cur_state)[0])
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
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)
        self.lg.debug("epsilon :" + str(self.epsilon))
        allowable_actions = self.env_grid.allowable_actions(cur_state)

        if np.random.random() < self.epsilon:
            self.lg.debug("R")
            if len(allowable_actions) == 0:
                print("??")
            actn = self.random_action(cur_state, allowable_actions)
        else:
            actn = self.actor_predict_action(cur_state, allowable_actions)
            if actn is None:
                self.lg.debug("R")
                actn = self.random_action(cur_state, allowable_actions)
            else:
                self.lg.debug("P")

        self.remember_action(cur_state, actn)
        self.steps_to_goal += 1
        return actn

    #
    # Track change of episode.
    #
    def new_episode(self):
        self.episode += 1
        self.train_on_new_episode = True
        self.lg.debug("************** Steps to Goal :" + str(self.steps_to_goal))
        self.steps_to_goal = 0

    #
    #
    #
    def get_num_episodes(self):
        return self.episode


#
# Return a grid as the environment. The aim is to learn the shortest
# path from the start cell to a goal cell, while avoiding the
# penalty cells (if any) . Each step has a negative cost, and it is this that drives
# the push for shortest path in terms of minimising the cost function.
#
# The action space is N,S,E,W=(0,1,2,3) shape (1,4)
# The state is the grid coordinates (x,y) shape (1,2)
#
step = SimpleGridOne.STEP
fire = SimpleGridOne.FIRE
blck = SimpleGridOne.BLCK
goal = SimpleGridOne.GOAL


def create_grid() -> SimpleGridOne:
    grid = [
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, fire, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, fire, step, goal, step, fire, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, fire, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step],
        [step, step, step, step, step, step, step, step, step, step, step, step, step, step, step]
    ]
    sg1 = SimpleGridOne(grid_id=1,
                        grid_map=grid,
                        respawn_type=SimpleGridOne.RESPAWN_RANDOM)
    return 15, 15, sg1


#
#
#
def main():
    lg = EnvironmentLogging("TestGridActorCritic2",
                            "TestGridActorCritic2.log",
                            logging.DEBUG).get_logger()

    sess = tf.Session()
    K.set_session(sess)
    n = 0
    r, c, env = create_grid()
    actor_critic = GridActorCritic(env, sess, lg, r, c)
    rdr = RenderSimpleGridOneQValues(num_cols=actor_critic.num_cols,
                                     num_rows=actor_critic.num_rows,
                                     plot_style=RenderSimpleGridOneQValues.PLOT_SURFACE,
                                     do_plot=True)

    env.reset()
    cur_state = env.state()

    while True:
        if env.episode_complete():
            env.reset()
            actor_critic.new_episode()
            cur_state = env.state()  # new_state
        else:
            action = actor_critic.select_action(cur_state)
            actor_critic.steps_to_goal += 1
            reward = env.execute_action(action)
            new_state = env.state()
            done = env.episode_complete()
            actor_critic.remember(cur_state, action, reward, new_state, done)
            lg.debug(":: " + str(cur_state) + " -> " + str(action) + " = " + str(reward))
            actor_critic.train()
            cur_state = env.state()  # new_state

            # Visualize.
            n += 1
            print("Iteration Number: " + str(n) + " of episode: " + str(actor_critic.get_num_episodes()))
            qgrid = np.zeros((actor_critic.num_rows, actor_critic.num_cols))
            if n % 10 == 0:
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
                    lg.debug(s)
                rdr.plot(qgrid)
                lg.debug('--------------------')


if __name__ == "__main__":
    main()
