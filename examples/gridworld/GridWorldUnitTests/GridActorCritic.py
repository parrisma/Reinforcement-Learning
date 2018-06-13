import random
import unittest
from typing import Tuple, List

import numpy as np

from examples.gridworld.Grid import Grid
from examples.gridworld.SimpleGridOne import SimpleGridOne
from examples.gridworld.TestRigs.GridWorldQValNNModel import GridWorldQValNNModel
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

        self.env_grid: Grid = grid
        self.lg = lg

        self.episode: int = 1

        self.batch_size: int = 32
        self.input_dim: int = 2
        self.num_actions: int = 4
        self.output_dim: int = self.num_actions
        self.num_rows: int = rows
        self.num_cols: int = cols

        self.learning_rate_0: float = float(1.0)
        self.learning_rate_decay: float = float(0.05)
        self.epsilon: float = 0.8  # exploration factor.
        self.epsilon_decay: float = .9995
        self.gamma: float = .8  # Discount Factor Applied to reward

        self.last_state: List[int] = None

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

    #
    # Extract a random sample from the replay memory, calculate the QVals and train
    # the critical deep NN model.
    #
    def _train_critic(self) -> bool:
        trained = False
        x, y = self._get_sample_batch()
        if x is not None:
            self.critic_model.train(x, y)
            trained = True
            self.lg.debug("Critic Trained")
        return trained

    #
    # Return the list of allowable actions minus the action that would return the agent
    # to the state it has just come from.
    #
    def allowed_actions_no_return(self, state: List[int]):
        aanr = list()
        allowable_actions = self.env_grid.allowable_actions(state)
        if self.last_state is None:
            return allowable_actions
        else:
            for action in allowable_actions:
                x, y = self.env_grid.coords_after_action(state[0], state[1], action)
                if x != self.last_state[0] and y != self.last_state[1]:
                    aanr.append(action)
        return aanr

    #
    # What is the prediction vector for the given state. This is adjusted to discount
    # dis-allowed actions.
    #
    # Will fail if the given state is the goal state as there are no allowable actions.
    #
    def _state_qval_prediction(self,
                               state) -> [np.float]:
        aa = self.allowed_actions_no_return(state)
        da = self.env_grid.disallowed_actions(aa)
        qvp = self._actor_prediction(state)[0]  # Actor estimate of QVals for next state. (after action)
        if len(aa) < self.num_actions:
            lv = np.min(qvp[aa])
            lv = lv - (0.01 * np.sign(lv) * lv)
            qvp[da] = lv  # suppress disallowed action by making qval less then smallest allowable qval / actn.
        return qvp

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
            allowable_actions = self.allowed_actions_no_return(cur_state)
            disallowed_actions = self.env_grid.disallowed_actions(allowable_actions)
            lr = self.learning_rate()

            qvn = self._actor_prediction(new_state)[0]  # Actor estimate of QVals for next state. (after action)
            qvp = np.max(qvn[allowable_actions])  # optimal return, can only be taken from allowable actions.
            qvp = self.gamma * qvp * lr  # Discounted max return from next state

            qvs = self._actor_prediction(cur_state)[0]  # Actor estimate of QVals for current state.
            if len(allowable_actions) < self.num_actions:
                lv = np.min(qvs[allowable_actions])
                lv = np.absolute(lv) * 1.01 * np.sign(lv)
                qvs[disallowed_actions] = lv  # suppress disallowed actions.
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
    # Set override actor predictor function (unit testing)
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
        actn = np.argmax(qvs[allowable_actions])
        return actn

    #
    # remember actions such that we can bias agent to prefer actions not yet seen or
    # lease visited.
    # explored from this state.
    #
    def remember_last_state(self, state: List[int]):
        self.last_state = state
        return

    #
    # Select a random action, but prefer an action not or least taken from
    # the current state (this should be replaced with another NN that learns
    # familiarity)
    #
    def random_action(self, allowable_actions: List[int]):
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
        allowable_actions = self.allowed_actions_no_return(cur_state)

        if np.random.random() < self.epsilon and not greedy:
            self.lg.debug("R")
            actn = self.random_action(allowable_actions)
        else:
            actn = self.actor_predict_action(cur_state, allowable_actions)
            if actn is None:
                self.lg.debug("R")
                actn = self.random_action(allowable_actions)
            else:
                self.lg.debug("P")

        self.remember_last_state(cur_state)
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


#
# Test Cases.
#
class TestGridActorCritic(unittest.TestCase):

    def test_0(self):
        self.assertTrue(True == True)
        return

    @classmethod
    def create_test_grid(cls) -> Tuple[int, int, SimpleGridOne]:
        r = 10
        c = 10
        grid = np.full((r, c), step)
        grid[2, 2] = goal
        sg1 = SimpleGridOne(grid_id=1,
                            grid_map=grid,
                            respawn_type=SimpleGridOne.RESPAWN_RANDOM)
        return r, c, sg1


#
# Execute the UnitTests.
#

if __name__ == "__main__":
    if True:
        tests = TestGridActorCritic()
        suite = unittest.TestLoader().loadTestsFromModule(tests)
        unittest.TextTestRunner().run(suite)
