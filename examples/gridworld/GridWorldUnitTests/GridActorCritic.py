import logging
import random
import unittest
from typing import Tuple

import numpy as np

from examples.gridworld.Grid import Grid
from examples.gridworld.SimpleGridOne import SimpleGridOne
from examples.gridworld.TestRigs.GridWorldQValNNModel import GridWorldQValNNModel
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.RareEventBiasReplayMemory import RareEventBiasReplayMemory


#
# Actor critic pattern to learn to find goals on a simple  2D grid.
#

class GridActorCritic:
    def __init__(self,
                 grid: Grid,
                 lg,
                 rows: int,
                 cols: int):
        self.env_grid = grid
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

        self.aux_actor_predictor = None

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
                                                num_epoch=3,
                                                lr_0=0.005,
                                                lr_min=0.001
                                                )

        self.critic_model = GridWorldQValNNModel(model_name="Critic",
                                                 input_dimension=self.input_dim,
                                                 num_actions=self.num_actions,
                                                 num_grid_cells=(self.num_rows * self.num_cols),
                                                 lg=self.lg,
                                                 batch_size=self.batch_size,
                                                 num_epoch=3,
                                                 lr_0=0.005,
                                                 lr_min=0.001
                                                 )
        return

    #
    # Return the learning rate based on number of learning's to date
    #
    def learning_rate(self):
        return self.learning_rate_0 / (1 + (self.episode * self.learning_rate_decay))

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
    # Distance in state space
    #
    @classmethod
    def _distance(cls, s1, s2) -> np.float:
        return np.sqrt(np.power(s1[0] - s2[0], 2) + np.power(s1[1] - s2[1], 2))

    #
    # What is the prediction vector for the given state. This is adjusted to discount
    # dis-allowed actions.
    #
    # Will fail if the given state is the goal state as there are no allowable actions.
    #
    def _state_qval_prediction(self,
                               state) -> [np.float]:
        aa = self.env_grid.allowable_actions(state)
        da = self.env_grid.disallowed_actions(aa)
        qvp = self._actor_prediction(state)[0]  # Actor estimate of QVals for next state. (after action)
        if len(aa) < self.num_actions:
            lv = np.min(qvp[aa])
            lv = lv - (0.01 * np.sign(lv) * lv)
            qvp[da] = lv  # suppress disallowed action by making qval less then smallest allowable qval / actn.
        return qvp

    #
    # Calculate the normalization factor
    #
    def _dist_norm_fact(self,
                        d: np.float,
                        mx: np.float,
                        mn: np.float) -> np.float:
        if mx == mn:
            if d == 0:
                return np.float(0)
            else:
                return np.float(1)
        else:
            return (d - mn) / (mx - mn)

    #
    # Search forward n steps for optimal next action. The predicted reward is
    # weighted by the distance in state space from current state with a view to
    # encourage exploration and penalise loops
    #
    def simulated_qval(self,
                       origin_state,
                       state,
                       gamma: np.float,
                       steps: int,
                       dmax: np.float = -np.finfo('d').max,
                       dmin: np.float = np.finfo('d').max,
                       ):

        qvp = self._state_qval_prediction(state)
        actn = np.argmax(qvp)
        reward = qvp[actn]
        next_state = self.env_grid.coords_after_action(state[0], state[1], actn)
        stp = steps - 1
        d = self._distance(origin_state, next_state)
        ndmax = max(dmax, d)
        ndmin = min(dmin, d)
        qv = 0
        if stp != 0:
            if self.env_grid.episode_complete(next_state):
                qv = reward
            else:
                qv, ndmax, ndmin = self.simulated_qval(origin_state, next_state, gamma * gamma, stp, ndmax, ndmin)

        qv = qv + reward - ((1.0 - self._dist_norm_fact(d, ndmax, ndmin)) * reward * np.sign(reward)) * gamma
        return qv, ndmax, ndmin

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
            aact = self.env_grid.allowable_actions(new_state)
            da = self.env_grid.disallowed_actions(aact)
            lr = self.learning_rate()

            qvns = self._state_qval_prediction(new_state)  # Actor estimate of QVals for
            qvnsa = np.argmax(qvns)
            if self.env_grid.coords_after_action(new_state[0], new_state[1], qvnsa):
                print('?')
                aact.remove(qvnsa)
                da.append(qvnsa)
                lv = np.min(qvns[aact])
                lv = lv - (0.01 * np.sign(lv) * lv)
                qv[da] = lv  # suppress disallowed action by making qval less then smallest allowable qval / actn.

            # current state.
            qvp = 0

            aact = self.env_grid.allowable_actions(cur_state)
            da = self.env_grid.disallowed_actions(aact)
            qvs = self._actor_prediction(cur_state)[0]  # Actor estimate of QVals for current state.
            if len(aact) < self.num_actions:
                lv = np.min(qvs[aact])
                lv = lv - (0.01 * np.sign(lv) * lv)
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
    # Set override actor predictor function.
    #
    def set_actor_predictor_function(self, predictor_func) -> None:
        self.aux_actor_predictor = predictor_func
        return

    #
    # Get actor prediction, if actor is not able to predict, predict random
    #
    def _actor_prediction(self, cur_state):
        if self.aux_actor_predictor is not None:
            return self.aux_actor_predictor(cur_state)

        z = np.zeros((1, 2))
        z[0] = cur_state
        p = self.actor_model.predict(z)
        if p is None:
            p = 0.5 * np.random.random_sample(self.num_actions) - 0.5
        return p

    #
    # Predict an allowable action.
    #
    def actor_predict_action(self, cur_state, allowable_actions) -> int:
        cur_state = np.array(cur_state).reshape((1, 2))  # Shape needed for NN
        qvs = self.actor_model.predict(cur_state)[0]
        da = self.env_grid.disallowed_actions(allowable_actions)
        if len(allowable_actions) < self.num_actions:
            lv = np.min(qvs[allowable_actions])
            if lv == 0:
                lv = -0.0001
            lv = lv - (0.01 * np.sign(lv) * lv)
            qvs[da] = lv  # suppress disallowed action by making qval less then smallest allowable qval / actn.
        actn = np.argmax(qvs)
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
    def select_action(self,
                      cur_state,
                      greedy: bool = False):
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
        self.lg.debug("epsilon :" + str(self.epsilon))
        allowable_actions = self.env_grid.allowable_actions(cur_state)

        if np.random.random() < self.epsilon and not greedy:
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
# Unit Tests.
#

step = SimpleGridOne.STEP
fire = SimpleGridOne.FIRE
blck = SimpleGridOne.BLCK
goal = SimpleGridOne.GOAL


def create_grid() -> Tuple[int, int, SimpleGridOne]:
    r = 5
    c = 5
    grid = np.full((r, c), step)
    grid[2, 2] = goal
    sg1 = SimpleGridOne(grid_id=1,
                        grid_map=grid,
                        respawn_type=SimpleGridOne.RESPAWN_RANDOM)
    return r, c, sg1


lg = EnvironmentLogging("UnitTestGridActorCritic",
                        "UnitTestGridActorCritic.log",
                        logging.DEBUG).get_logger()


#
# The numbers in the test cases are just simulations of what the deep NN will predict
# for the q-vals of the given state. They are all uniform to keep the tests predictable
# but in practice they will be varied approximations.
#
@unittest.skip("This is a support class only.")
class TestPredictorFactory:
    def __init__(self):
        self.i = 0
        self.cases = None

    #
    # Dummy predictor
    #
    def predict(self, _) -> [np.float]:
        c = self.cases[self.i]
        self.i += 1
        return np.reshape(c, (1, 4))

    #
    # North - South loop
    #
    def north_south_one_step_loop(self):
        self.cases = [[-0.25, -0.2, -0.25, -0.25]  # S
                      ]
        self.i = 0

    #
    # North - North, one step away
    #
    def north_north_one_step_loop(self):
        self.cases = [[-0.2, -0.25, -0.25, -0.25]  # N
                      ]
        self.i = 0

    #
    # Three Step Loop assume S' = N (E,S,W)
    #
    def three_step_loop(self):
        self.cases = [[-0.25, -0.25, -0.2, -0.25],  # E
                      [-0.25, -0.2, -0.25, -0.25],  # S
                      [-0.25, -0.25, -0.25, -0.2]  # W
                      ]
        self.i = 0

    #
    # Three Step Loop assume S' = N (N,W,S)
    #
    def three_step_away(self):
        self.cases = [[-0.2, -0.25, -0.25, -0.25],  # N
                      [-0.25, -0.25, -0.25, -0.2],  # W
                      [-0.25, -0.2, -0.25, -0.25]  # S
                      ]
        self.i = 0

    #
    # Num steps to test case
    #
    def steps(self) -> int:
        return len(self.cases)

    #
    # Deep North - South loop
    #
    def deep_north_south_step_loop(self):
        self.cases = [[-0.25, -0.2, -0.25, -0.25],  # S
                      [-0.2, -0.25, -0.25, -0.25],  # N
                      [-0.25, -0.2, -0.25, -0.25],  # S
                      [-0.2, -0.25, -0.25, -0.25],  # N
                      [-0.25, -0.2, -0.25, -0.25],  # S
                      [-0.2, -0.25, -0.25, -0.25],  # N
                      [-0.25, -0.2, -0.25, -0.25],  # S
                      [-0.2, -0.25, -0.25, -0.25]  # N
                      ]
        self.i = 0

    #
    # Deep North - South loop
    #
    def six_steps(self):
        self.cases = [[-0.2, -0.25, -0.25, -0.25],  # N
                      [-0.2, -0.25, -0.25, -0.25],  # N
                      [-0.2, -0.25, -0.25, -0.25],  # N
                      [-0.25, -0.25, -0.2, -0.25],  # E
                      [-0.25, -0.2, -0.25, -0.25],  # S
                      [-0.25, -0.2, -0.25, -0.25],  # S
                      [-0.25, -0.2, -0.25, -0.25],  # S
                      [-0.25, -0.25, -0.25, -0.2]  # W
                      ]
        self.i = 0

    #
    # Three Steps via goal state at 2,2 : should stop on 2nd move
    #
    def towards_goal_state(self):
        self.cases = [[-0.2, -0.25, -0.25, -0.25],  # N
                      [-0.25, -0.25, -0.25, 1.0],  # W (Goal State)
                      [-0.25, -0.22, -0.25, -0.2]  # W (should never get here)
                      ]
        self.i = 0


#
# Test Cases.
#
class TestGridActorCritic(unittest.TestCase):

    def __run_cases(self, pf, cases):
        res = [0, 0]
        i = 0
        for t in cases:
            t()
            r, c, env = create_grid()
            actor_critic = GridActorCritic(env, lg, r, c)
            actor_critic.set_actor_predictor_function(pf.predict)
            res[i], _, _ = actor_critic.simulated_qval(origin_state=[4, 3],
                                                       state=[3, 3],
                                                       gamma=0.8,
                                                       steps=pf.steps())
            i == 1
        self.assertTrue(res[0] < res[1])
        return

    def test_0(self):
        pf = TestPredictorFactory()
        tst = [pf.north_south_one_step_loop, pf.north_north_one_step_loop]
        self.__run_cases(pf, tst)
        return

    def test_1(self):
        pf = TestPredictorFactory()
        tst = [pf.three_step_loop, pf.three_step_away]
        self.__run_cases(pf, tst)
        return

    def test_2(self):
        pf = TestPredictorFactory()
        tst = [pf.deep_north_south_step_loop, pf.six_steps]
        self.__run_cases(pf, tst)
        return

    def test_goal_state(self):
        pf = TestPredictorFactory()
        pf.towards_goal_state()
        r, c, env = create_grid()
        actor_critic = GridActorCritic(env, lg, r, c)
        actor_critic.set_actor_predictor_function(pf.predict)
        vs, _, _ = actor_critic.simulated_qval(origin_state=[4, 3],
                                               state=[3, 3],
                                               gamma=0.8,
                                               steps=pf.steps())
        self.assertAlmostEqual(vs, 1.64, 5)
        return


#
# Execute the UnitTests.
#

if __name__ == "__main__":
    if True:
        tests = TestGridActorCritic()
        suite = unittest.TestLoader().loadTestsFromModule(tests)
        unittest.TextTestRunner().run(suite)
