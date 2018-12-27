import logging
import random
import unittest

import numpy as np

from examples.tictactoe.TicTacToe import TicTacToe
from examples.tictactoe.TicTacToeAgent import TicTacToeAgent
from examples.tictactoe.TicTacToeState import TicTacToeState
from examples.tictactoe.TicTacToeTests.TestState import TestState
from reflrn.ActorCriticPolicy import ActorCriticPolicy
from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.GeneralModelParams import GeneralModelParams
from reflrn.Interface.ModelParams import ModelParams
from reflrn.PureRandomExploration import PureRandomExploration
from .TestAgent import TestAgent

lg = EnvironmentLogging("TestActorCriticPolicy", "TestActorCriticPolicy.log", logging.INFO).get_logger()


class TestActorCriticPolicy(unittest.TestCase):

    #
    # Try to ensure tests are repeatable
    #
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        random.seed(42)

    #
    # Just construct a policy and ensure no exceptions are thrown
    #
    def test_simple_bootstrap(self):
        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        ttt = TicTacToe(agent_x, agent_o, None)
        acp = ActorCriticPolicy(env=ttt, lg=lg)
        self.assertIsNotNone(acp)

    #
    # Train a policy on generated state/action data and see if policy can predict correct
    # action. The test data has same reward for every action; so this should drive the policy
    # to predict the generated action associated with the given state.
    #
    def test_actor_critic_policy(self):
        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        ttt = TicTacToe(agent_x, agent_o, None)

        pp = GeneralModelParams([[ModelParams.epsilon, float(1)],
                                 [ModelParams.epsilon_decay, float(0)],
                                 [ModelParams.verbose, int(2)],
                                 [ModelParams.train_every, int(100)],
                                 [ModelParams.num_states, int(50)]
                                 ])

        acp = ActorCriticPolicy(env=ttt,
                                policy_params=pp,
                                lg=lg)

        ns = None
        dt = self.get_data()
        ag = agent_o
        i = 0
        el = np.random.randint(5, 9)  # select a random episode length between 5 and 9 plays
        for k in dt.keys():
            st = TestState(st=(dt[k])[0],
                           shp=(3, 3))
            a = (dt[k])[1]
            if ns is None:
                ns = st
            else:

                acp.update_policy(agent_name=ag.name(),
                                  state=st,
                                  next_state=ns,
                                  action=a,
                                  reward=float(1),
                                  episode_complete=(i == el))
                if ag == agent_o:
                    ag = agent_x
                else:
                    ag = agent_o

                ns = st
                if i >= el:
                    i = 0
                    el = np.random.randint(5, 9)
                else:
                    i += 1

        # Now check prediction accuracy
        #
        ok = 0
        for k in dt.keys():
            pred_actn = acp.greedy_action(state=TestState(st=(dt[k])[0], shp=(1, 9)))
            if (dt[k])[1] == pred_actn:
                ok += 1
        self.assertGreater((ok / len(dt)), 0.95)
        return

    #
    # Test the discounting (reward attribution) by playing many runs of the same game
    #
    def test_discounting(self):

        iterations = 1000

        acp_x = ActorCriticPolicy(policy_params=self.default_model_params(num_states=8),
                                  lg=lg)
        acp_x.static_test_action_list = (0, 2, 6, 5)

        agent_x = TicTacToeAgent(1,
                                 "X",
                                 acp_x,
                                 epsilon_greedy=0,  # has no impact as static actions defined
                                 exploration_play=PureRandomExploration(),  # has no impact as static actions defined
                                 lg=lg)

        acp_o = ActorCriticPolicy(policy_params=self.default_model_params(num_states=8),
                                  lg=lg)
        acp_o.static_test_action_list = (8, 1, 7, 4)

        agent_o = TicTacToeAgent(-1,
                                 "O",
                                 acp_o,
                                 epsilon_greedy=0,  # has no impact as static actions defined
                                 exploration_play=PureRandomExploration(),  # has no impact as static actions defined
                                 lg=lg)

        game = TicTacToe(agent_x, agent_o, lg)
        acp_o.link_to_env(game)
        acp_x.link_to_env(game)
        acp_o.explain = False
        acp_x.explain = False

        lqv = 0
        for j in range(0, iterations):
            game.run(100)

            s2 = TicTacToeState(np.array((1,
                                          -1,
                                          1,
                                          np.nan,
                                          np.nan,
                                          1,
                                          1,
                                          -1,
                                          -1)),
                                agent_x, agent_o)

            qv = acp_x._ActorCriticPolicy__critic_prediction(s2)
            # lg.info(qv)
            qvm = max(qv)
            if qvm != lqv:
                lg.info(qvm)
            lqv = qvm

        return

    #
    # Create 50 random state to action mappings and then create a training set of 3000
    # by extracting samples at random (with repeats) from the 50 state action mappings.
    # Each state will only map to only one action.
    #
    # This means model will see many repeats of the same state/action combo.
    #
    def get_data(self) -> dict:
        num_samples = 50
        samples = dict()
        np.random.seed(42)  # Thanks Douglas.
        while len(samples) < num_samples:
            n = np.random.randint(0, (2 ** 9) - 1)
            if n not in samples:
                st = self.binary_as_one_hot(np.binary_repr(n, width=9))
                a = np.random.randint(0, 9)
                samples[n] = [st, a]

        dt = dict()
        skeys = list(samples.keys())
        for i in range(0, 3000):
            ky = skeys[np.random.randint(0, 19)]
            dt[i] = samples[ky]

        return dt

    #
    # Binary string to numpy array one hot encoding as float
    #
    @classmethod
    def binary_as_one_hot(cls,
                          bstr: str) -> np.array:
        npa = np.array(list(map(float, bstr)))
        return npa

    #
    # Model parameter defaults for testing
    #
    @classmethod
    def default_model_params(cls,
                             num_states: int = 5500) -> GeneralModelParams:
        return GeneralModelParams([[ModelParams.epsilon, float(1)],
                                   [ModelParams.epsilon_decay, float(0)],
                                   [ModelParams.num_actions, int(9)],
                                   [ModelParams.model_file_name, 'TicTacToe-ActorCritic'],
                                   [ModelParams.verbose, int(0)],
                                   [ModelParams.num_states, num_states]
                                   ])


#
# Execute the Test Actor Critic Policy Unit Test Suite.
#
if __name__ == "__main__":
    tests = TestActorCriticPolicy()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
