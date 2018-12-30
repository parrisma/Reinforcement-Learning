import logging
import random
import unittest
from typing import List

import numpy as np

from examples.gridworld.Grid import Grid
from examples.gridworld.SimpleGridOne import SimpleGridOne
from examples.gridworld.GridWorldQValNNModel import GridWorldQValNNModel
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
        self.learning_rate_decay = float(0.02)
        self.epsilon = 0.8  # exploration factor.
        self.epsilon_decay = .9995
        self.gamma = .8  # Discount Factor Applied to reward

        self.steps_to_goal = 0
        self.train_on_new_episode = False

        self.aux_actor_predictor = None

        self.__training = True  # by default we train actor/critic as we take actions

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
        rw, cl = self._get_sample_batch()
        if rw is not None:
            self.critic_model.train(rw, cl)
            trained = True
            self.lg.debug("Critic Trained")
        return trained

    #
    # Return the list of allowable actions minus the action that would return the agent
    # to the curr_coords it has just come from.
    #
    def allowed_actions_no_return(self,
                                  state: List[int],
                                  last_state: List[int] = None):
        aanr = list()
        allowable_actions = self.env_grid.allowable_actions(state)
        if last_state is None:
            return allowable_actions
        else:
            for action in allowable_actions:
                rw, cl = self.env_grid.coords_after_action(state[Grid.ROW], state[Grid.COL], action)
                if not (rw == last_state[Grid.ROW] and cl == last_state[Grid.COL]):
                    aanr.append(action)
        return aanr

    #
    # Get actor prediction, if actor is not able to predict, predict random
    #
    def _actor_prediction(self,
                          curr_state: List[int]):
        if self.aux_actor_predictor is not None:
            return self.aux_actor_predictor(curr_state)

        st = np.array(curr_state).reshape((1, 2))  # Shape needed for NN
        p = self.actor_model.predict(st)[0]  # Can predict even if model is not trained, just predicts random.
        return p

    #
    # Get the last state from memory with respect to the given state
    #
    def _get_last_state(self, curr_state: List[int]) -> List[int]:
        lst_state = None
        if not self.env_grid.episode_complete(curr_state):
            lst_state = (self.replay_memory.get_last_memory(curr_state))
            if lst_state is not None:
                lst_state = lst_state[0]
        return lst_state

    #
    # What is the optimal QVal prediction for next curr_coords S'. Return zero if next curr_coords
    # is the end of the episode.
    #
    def _next_state_qval_prediction(self,
                                    new_state: List[int],
                                    last_state: List[int]) -> float:
        qvp = 0
        if not self.env_grid.episode_complete(new_state):
            qvn = self._actor_prediction(curr_state=new_state)
            allowable_actions = self.allowed_actions_no_return(state=new_state, last_state=last_state)
            qvp = self.gamma * np.max(qvn[allowable_actions])  # Discounted max return from next curr_coords
        return np.float(qvp)

    #
    # What is the qvalue (optimal) prediction given current curr_coords (state S)
    #
    def _curr_state_qval_prediction(self, curr_state: List[int]) -> List[np.float]:
        qvs = np.zeros(self.num_actions)
        if not self.env_grid.episode_complete(curr_state):
            qvs = self._actor_prediction(curr_state)
        return qvs

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
            lr = self.learning_rate()
            qvp = self._next_state_qval_prediction(new_state, cur_state)
            qvs = self._curr_state_qval_prediction(cur_state)

            qv = qvs[action]
            qv = (qv * (1 - lr)) + (lr * (reward + qvp))  # updated expectation of current curr_coords/action
            qvs[action] = qv

            x[i] = cur_state
            y[i] = qvs
            i += 1

        return x, y

    #
    # Update actor every 5 episodes (goal states reached)
    #
    def train(self):
        if not self.__training:
            return

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
    # Predict an action, this may not be an allowed action at the given location.
    #
    def actor_predict_action(self, cur_state) -> int:
        qvs = self._actor_prediction(cur_state)
        actn = np.argmax(qvs)
        return actn

    #
    # Select a random action, but prefer an action not or least taken from
    # the current curr_coords (this should be replaced with another NN that learns
    # familiarity)
    #
    @classmethod
    def random_action(cls, allowable_actions: List[int]):
        actn = random.choice(allowable_actions)
        return actn

    #
    # Update and Return the exploration factor epsilon
    #
    def __update_epsilon(self) -> float:
        if self.__training:
            self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
        return self.epsilon

    #
    # Make a greedy (random) action based on current value (decaying) of epsilon
    # else make an action based on prediction the NN inside the actor.
    #
    def select_action(self,
                      cur_state,
                      greedy: bool = False):
        self.__update_epsilon()
        lst_state = (self.replay_memory.get_last_memory(cur_state))
        if lst_state is not None:
            lst_state = lst_state[0]
        allowable_actions = self.allowed_actions_no_return(cur_state, lst_state)

        if True:  # np.random.random() < self.epsilon and not greedy:
            self.lg.debug("R")
            actn = self.random_action(allowable_actions)
        else:
            actn = self.actor_predict_action(cur_state)
            if actn not in allowable_actions:
                self.lg.debug("R'")
                actn = self.random_action(allowable_actions)
            else:
                self.lg.debug("P")
        return actn

    #
    # Return the Q Value Grid as predicted by the current state of the actor.
    #
    def qvalue_grid(self,
                    average: bool = True) -> List[np.float]:
        qgrid = np.zeros((self.num_rows, self.num_cols))
        for rw in range(0, self.num_rows):
            for cl in range(0, self.num_cols):
                st = np.array([rw, cl]).reshape((1, self.input_dim))
                q_vals = self.critic_model.predict(st)[0]
                for actn in self.allowed_actions_no_return(state=st[0]):
                    if average:
                        r, c = SimpleGridOne.coords_after_action(rw, cl, actn)
                        if r >= 0 and c >= 0 and r < self.num_rows and c < self.num_cols:
                            if qgrid[r][c] == np.float(0):
                                qgrid[r][c] = q_vals[actn]
                            else:
                                qgrid[r][c] += q_vals[actn]
                                qgrid[r][c] /= np.float(2)
                    else:
                        qgrid[rw][cl] = np.max(q_vals)
        return qgrid

    #
    # Track change of episode.
    #
    def new_episode(self):
        if self.__training:
            self.episode += 1
        self.train_on_new_episode = True
        self.lg.debug("************** Steps to Goal :" + str(self.steps_to_goal))
        self.steps_to_goal = 0

    #
    # How many episodes have passed = number of goal states reached.
    #
    def get_num_episodes(self):
        return self.episode

    #
    # Set training mode on/off
    #
    def training_mode_on(self) -> None:
        self.__training = True
        return

    def training_mode_off(self) -> None:
        self.__training = False
        return


#
# Unit Tests.
#

step = SimpleGridOne.STEP
fire = SimpleGridOne.FIRE
blck = SimpleGridOne.BLCK
goal = SimpleGridOne.GOAL

lg = EnvironmentLogging("GridActorCriticUnitTest",
                        "GridActorCriticUnitTest.log",
                        logging.DEBUG).get_logger()


#
# Test Cases.
#
class TestGridActorCritic(unittest.TestCase):

    #
    # Test, all possible moves on 3 by 3 grid
    #
    def test_0(self):
        grid = [
            [step, step, step],
            [step, step, step],
            [step, step, step]
        ]
        sg0 = SimpleGridOne(0,
                            grid,
                            [1, 1])

        actor_critic = GridActorCritic(sg0, lg, 3, 3)

        test_cases = [[(0, 0), 2, [SimpleGridOne.SOUTH, SimpleGridOne.EAST]],
                      [(0, 1), 3, [SimpleGridOne.WEST, SimpleGridOne.SOUTH, SimpleGridOne.EAST]],
                      [(0, 2), 2, [SimpleGridOne.WEST, SimpleGridOne.SOUTH]],
                      [(1, 0), 3, [SimpleGridOne.NORTH, SimpleGridOne.EAST, SimpleGridOne.SOUTH]],
                      [(1, 1), 4, [SimpleGridOne.NORTH, SimpleGridOne.EAST, SimpleGridOne.SOUTH, SimpleGridOne.EAST]],
                      [(1, 2), 3, [SimpleGridOne.WEST, SimpleGridOne.NORTH, SimpleGridOne.SOUTH]],
                      [(2, 0), 2, [SimpleGridOne.NORTH, SimpleGridOne.EAST]],
                      [(2, 1), 3, [SimpleGridOne.NORTH, SimpleGridOne.WEST, SimpleGridOne.EAST]],
                      [(2, 2), 2, [SimpleGridOne.WEST, SimpleGridOne.NORTH]]
                      ]

        # Test before first action is taken.
        for coords, ln, moves in test_cases:
            aac = actor_critic.allowed_actions_no_return(state=coords, last_state=None)
            self.assertEqual(len(aac), ln)
            for mv in moves:
                self.assertTrue(mv in aac)

        # Move south, this should reject a move North even though it is a technically
        # possible move.
        last_state = sg0.curr_coords()
        sg0.execute_action(SimpleGridOne.SOUTH)
        aac = actor_critic.allowed_actions_no_return(state=sg0.curr_coords(), last_state=last_state)
        self.assertEqual(SimpleGridOne.WEST in aac, True)
        self.assertEqual(SimpleGridOne.EAST in aac, True)

        return


#
# Execute the UnitTests.
#

if __name__ == "__main__":
    tests = TestGridActorCritic()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
