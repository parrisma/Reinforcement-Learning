import logging
from random import randint

import numpy as np

from reflrn.EvaluationException import EvaluationException
from reflrn.FixedGames import FixedGames
from reflrn.Interface.Policy import Policy
from reflrn.Interface.RenderQVals import RenderQVals
from reflrn.Interface.State import State
from reflrn.RandomPolicy import RandomPolicy
from reflrn.TemporalDifferenceQValPolicyPersistance import TemporalDifferenceQValPolicyPersistance


class TemporalDifferenceQValPolicy(Policy):
    #
    # Learning is for all agents of this *type* so q values are at class level, and all
    # methods that act on q values are class methods.
    #
    # ToDo: q_vals should not be at class level, should pass in the q_val dicts so it can be shared only if required
    #
    __q_values = None  # key: Agent id + State & Value: (dictionary of key: action -> Value: Q value)
    __n = 0  # number of learning events
    __learning_rate_0 = float(1.0)
    __discount_factor = float(0.8)
    __learning_rate_decay = float(0.05)
    __fixed_games = None  # a class that allows canned games to be played
    __rand_qval_init = True

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self,
                 lg: logging,
                 filename: str = None,
                 fixed_games: FixedGames = None,
                 load_qval_file: bool = False,
                 manage_qval_file: bool = False,
                 save_every: int = 5000,
                 q_val_render: RenderQVals = None):
        self.__lg = lg
        self.__filename = filename
        self.__persistance = TemporalDifferenceQValPolicyPersistance()
        self.__persistance.enable_csv_file_save()
        self.__fixed_games = fixed_games
        self.__manage_qval_file = manage_qval_file
        self.__save_every = save_every  # how often do we dump down the q value file if save is enabled.
        self.__q_val_render = q_val_render
        self.__fallback_policy = RandomPolicy(prefer_new=True)
        self.__frame_id = 0

        if load_qval_file:
            try:
                (TemporalDifferenceQValPolicy.__q_values,
                 TemporalDifferenceQValPolicy.__n,
                 TemporalDifferenceQValPolicy.__learning_rate_0,
                 TemporalDifferenceQValPolicy.__discount_factor,
                 TemporalDifferenceQValPolicy.__learning_rate_decay) \
                    = self.__persistance.load(filename)
            except RuntimeError:
                pass  # File does not exist, keep class level defaults
        return

    #
    # Return the learning rate based on number of learning's to date
    #
    @classmethod
    def __q_learning_rate(cls, n: int):
        return cls.__learning_rate_0 / (1 + (n * cls.__learning_rate_decay))

    #
    # Manage q value store
    #
    @classmethod
    def __manage_q_val_store(cls, q_val_state_name: str, action: int):
        if cls.__q_values is None:
            cls.__q_values = dict()
            cls.__n = 0

        if q_val_state_name not in cls.__q_values:
            cls.__q_values[q_val_state_name] = dict()

        if action not in cls.__q_values[q_val_state_name]:
            cls.__q_values[q_val_state_name][action] = cls.__init_qval(cls.__rand_qval_init)

        return

    #
    # Get the given q value for the given agent, state and action
    #
    @classmethod
    def __get_q_value(cls, state: State, action: int) -> float:
        state_name = state.state_as_string()
        cls.__manage_q_val_store(state.state_as_string(), action)
        return cls.__q_values[state_name][action]

    #
    # Set the q value for the given agent, state and action
    #
    @classmethod
    def __set_q_value(cls, state: State, action: int, q_value: float) -> None:
        state_name = state.state_as_string()
        cls.__manage_q_val_store(state_name, action)
        cls.__q_values[state_name][action] = q_value

    #
    # Use temporal difference methods to keep q values for the given state/action plays.
    #
    # prev_state : the previous state for this Agent; None if no previous state
    # prev_action : the previous action of this agent; has no meaning is prev_state = None
    # state : current state of the environment *after* the given action was played
    # action : the action played by this agent that moved the state to the state passed
    # reward : the reward associated with the given state/action pair.
    # ToDo
    def update_policy(self,
                      agent_name: str,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool) -> None:

        self.__lg.debug(
            str(
                self.__frame_id) + ":" + agent_name + " : " + state.state_as_string() + " : " + next_state.state_as_string() + " : " + str(
                action))

        lgm = self.vals_and_actions_as_str(state)
        if episode_complete:
            self.__lg.debug(lgm)
            self.__frame_id += 1
            if self.__manage_qval_file:  # Save Q Vals At End Of Every Episode
                self.__save()

        # Update master count of policy learning events
        TemporalDifferenceQValPolicy.__n += 1

        lr = TemporalDifferenceQValPolicy.__q_learning_rate(self.__frame_id)

        # Establish the max (optimal) outcome taken from the target state.
        #
        qvs, actn = TemporalDifferenceQValPolicy.__get_q_vals_as_np_array(next_state)
        ou = TemporalDifferenceQValPolicy.__greedy_outcome(qvs)
        qvp = self.__discount_factor * ou * lr

        # Update current state to reflect the reward
        qv = TemporalDifferenceQValPolicy.__get_q_value(state, action)
        qv = (qv * (1 - lr)) + (lr * reward) + qvp
        TemporalDifferenceQValPolicy.__set_q_value(state, action, qv)

        return

    #
    # The optimal action is to take the largest positive gain or the smallest
    # loss. => Maximise Gain & Minimise Loss
    #
    @classmethod
    def __greedy_outcome(cls, q_values: np.array) -> np.float:
        if q_values is not None and q_values.size > 0:
            return np.max(q_values)
        else:
            return np.float(0)

    #
    # get_memories_by_type q values and associated actions as numpy array
    #
    @classmethod
    def __get_q_vals_as_np_array(cls, state: State) -> np.array:
        q_values = None
        q_actions = None

        # If there are no Q values learned yet we cannot predict a greedy action.
        if cls.__q_values is not None:
            state_name = state.state_as_string()

            if state_name in cls.__q_values:
                sz = len(cls.__q_values[state_name])
                q_values = np.full(sz, np.nan)
                q_actions = np.array(sorted(list(cls.__q_values[state_name].keys())))
                i = 0
                for actn in q_actions:
                    q_values[i] = cls.__q_values[state_name][actn]
                    i += 1

        return q_values, q_actions

    #
    # Greedy action; return the action that has the strongest Q value or if there is more
    # than one q value with the same strength, return an arbitrary action from those with
    # equal strength.
    #
    def select_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:

        if self.__fixed_games is not None:
            return self.__fixed_games.next_action()

        qvs, actions = self.__get_q_vals_as_np_array(state)

        # If no Q Values to drive direction then use the fallback policy.
        # the fallback policy will always return an action.
        if qvs is None:
            return self.__fallback_policy.select_action(agent_name, state, possible_actions)

        ou = TemporalDifferenceQValPolicy.__greedy_outcome(qvs)
        greedy_actions = list()
        for v, a in np.vstack([qvs, actions]).T:
            if v == ou:
                if a in possible_actions:
                    greedy_actions.append(int(a))
        if len(greedy_actions) == 0:
            raise EvaluationException("No Q Values mapping to possible actions, cannot select greedy action")

        return greedy_actions[randint(0, len(greedy_actions) - 1)]

    #
    # Save with class default filename.
    #
    def __save(self):
        self.save(self.__filename)

    #
    # FileName, return the given file name of the one set as default during
    # class construction
    #
    def file_name(self, filename: str) -> str:
        fn = filename
        if fn is None or len(fn) == 0:
            fn = self.__filename
        return fn

    #
    # Export the current policy to the given file name
    #
    def save(self, filename: str = None):
        fn = self.file_name(filename)
        if fn is not None and len(fn) > 0:
            self.__persistance.save(TemporalDifferenceQValPolicy.__q_values,
                                    TemporalDifferenceQValPolicy.__n,
                                    TemporalDifferenceQValPolicy.__learning_rate_0,
                                    TemporalDifferenceQValPolicy.__discount_factor,
                                    TemporalDifferenceQValPolicy.__learning_rate_decay,
                                    fn)
        else:
            raise FileNotFoundError("File name for TemporalDifferencePolicy save does not exist: [" & fn & "]")

        return

    #
    # Import the current policy to the given file name
    #
    def load(self, filename: str = None):
        fn = self.file_name(filename)
        if fn is not None and len(fn) > 0:
            (TemporalDifferenceQValPolicy.__q_values,
             TemporalDifferenceQValPolicy.__n,
             TemporalDifferenceQValPolicy.__learning_rate_0,
             TemporalDifferenceQValPolicy.__discount_factor,
             TemporalDifferenceQValPolicy.__learning_rate_decay) \
                = self.__persistance.load(filename)
        else:
            raise FileNotFoundError("File name for TemporalDifferencePolicy Load does not exist: [" & fn & "]")

        return (TemporalDifferenceQValPolicy.__q_values,
                TemporalDifferenceQValPolicy.__n,
                TemporalDifferenceQValPolicy.__learning_rate_0,
                TemporalDifferenceQValPolicy.__discount_factor,
                TemporalDifferenceQValPolicy.__learning_rate_decay)

    #
    # Q Values as a string (in grid form). This is just a visual debugger so it is
    # possible to see what q values are being selected from in the way that they
    # relate to the board. (3 x 3)
    #
    def vals_and_actions_as_str(self, state: State) -> str:
        return self.__q_val_render.render(state, self.__q_values)

    #
    # Log state
    #
    def __log_state(self):
        pass

    #
    # Q Value Initialize
    #
    @classmethod
    def __init_qval(cls, rand_init: bool = True) -> np.float:
        if rand_init:
            return np.random.uniform(-1, 1)
        else:
            return np.float(0)
