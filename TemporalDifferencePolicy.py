import numpy as np
import logging
from Policy import Policy
from State import State
from TemporalDifferencePolicyPersistance import TemporalDifferencePolicyPersistance
from EvaluationException import EvaluationExcpetion
from random import randint
from FixedGames import FixedGames
from typing import Tuple


class TemporalDifferencePolicy(Policy):

    #
    # Learning is for all agents of this *type* so q values are at class level, and all
    # methods that act on q values are class methods.
    #
    __q_values = None  # key: Agent id + State & Value: (dictionary of key: action -> Value: Q value)
    __n = 0  # number of learning events
    __learning_rate_0 = float(0.05)
    __discount_factor = float(0.8)
    __learning_rate_decay = float(0.001)
    __fixed_games = None  # a class that allows canned games to be played

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self, lg: logging, filename: str="", fixed_games: FixedGames=None, load_file: bool=False):
        self.__lg = lg
        self.__filename = filename
        self.__persistance = TemporalDifferencePolicyPersistance()
        self.__fixed_games = fixed_games

        if load_file:
            try:
                (TemporalDifferencePolicy.__q_values,
                 TemporalDifferencePolicy.__n,
                 TemporalDifferencePolicy.__learning_rate_0,
                 TemporalDifferencePolicy.__discount_factor,
                 TemporalDifferencePolicy.__learning_rate_decay) \
                    = self.__persistance.load(filename)
            except RuntimeError:
                pass  # File does not exist, keep class level defaults
        return

    #
    # Return the learning rate based on number of learning's to date
    #
    @classmethod
    def __q_learning_rate(cls):
        return cls.__learning_rate_0 / (1 + (cls.__n * cls.__learning_rate_decay))

    @classmethod
    def __q_value_state_name(cls, agent_name: str, state: State) -> str:
        return state.state_as_string()
        #  return agent_name + ":" + state.state_as_string()

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
            cls.__q_values[q_val_state_name][action] = float(0)

        return

    #
    # Get the given q value for the given agent, state and action
    #
    @classmethod
    def __get_q_value(cls, agent_name: str, state: State, action: int):
        state_name = cls.__q_value_state_name(agent_name, state)
        cls.__manage_q_val_store(state_name, action)
        return cls.__q_values[state_name][action]

    #
    # Set the q value for the given agent, state and action
    #
    @classmethod
    def __set_q_value(cls, agent_name: str, state: State, action: int, q_value: float):
        state_name = cls.__q_value_state_name(agent_name, state)
        cls.__manage_q_val_store(state_name, action)
        cls.__q_values[state_name][action] = q_value

    #
    # get q values and associated actions as numpy array
    #
    @classmethod
    def __get_q_vals_as_np_array(cls, agent_name: str, state: State) -> np.array:
        q_values = None
        q_actions = None

        # If there are no Q values learned yet we cannot predict a greedy action.
        if cls.__q_values is not None:
            state_name = cls.__q_value_state_name(agent_name, state)

            if state_name in cls.__q_values:
                sz = len(cls.__q_values[state_name])
                q_values = np.full(sz, np.nan)
                q_actions = np.full(sz, np.int(0))
                i = 0
                for k, v in cls.__q_values[state_name].items():
                    q_values[i] = v
                    q_actions[i] = np.int(k)
                    i += 1

        return q_values, q_actions

    #
    # Use temporal difference methods to keep q values for the given state/action plays.
    #
    # prev_state : the previous state for this Agent; None if no previous state
    # prev_action : the previous action of this agent; has no meaning is prev_state = None
    # state : current state of the environment *after* the given action was played
    # action : the action played by this agent that moved the state to the state passed
    # reward : the reward associated with the given state/action pair.
    #
    def update_policy(self, agent_name: str, prev_state: State, prev_action: int, state: State, action: int, reward: float):

        # Update master count of policy learning events
        TemporalDifferencePolicy.__n += 1
        if self.__n % 10 == 0:
            self.__save()

        # Update current state to reflect the reward
        qv = TemporalDifferencePolicy.__get_q_value(agent_name, state, action)
        lr = TemporalDifferencePolicy.__q_learning_rate()
        qv = (qv * (1-lr)) + (lr*reward)
        TemporalDifferencePolicy.__set_q_value(agent_name, state, action, qv)

        # discount the reward to prior state so we can establish reward attribution path
        # through state space.
        #
        if prev_state is not None:
            qvp = TemporalDifferencePolicy.__get_q_value(agent_name, prev_state, prev_action)
            qvs, actn = TemporalDifferencePolicy.__get_q_vals_as_np_array(agent_name, state)
            ou = TemporalDifferencePolicy.__greedy_outcome(qvs)
            qvp += self.__discount_factor * ou * lr
            TemporalDifferencePolicy.__set_q_value(agent_name, prev_state, prev_action, qvp)
        return

    #
    # The optimal action is to take the largest positive gain or the smallest
    # loss. => Maximise Gain & Minimise Loss
    #
    @classmethod
    def __greedy_outcome(cls, q_values: np.array) -> float:
        if len(q_values) > 0:
            stand_to_win = q_values[q_values >= 0]
            stand_to_lose = q_values[q_values < 0]
            if stand_to_win.size > 0:
                return np.max(stand_to_win)
            else:
                if stand_to_lose.size > 0:
                    return np.max(stand_to_lose)
                else:
                    return np.nan
        else:
            return np.nan

    #
    # Greedy action; return the action that has the strongest Q value or if there is more
    # than one q value with the same strength, return an arbitrary action from those with
    # equal strength.
    #
    def greedy_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:

        if self.__fixed_games is not None:
            return self.__fixed_games.next_action()

        qvs, actions = TemporalDifferencePolicy.__get_q_vals_as_np_array(agent_name, state)
        self.__lg.debug(self.vals_and_actions_as_str(qvs, actions))
        if qvs is None:
            raise EvaluationExcpetion("No Q Values with which to select greedy action")
        ou = TemporalDifferencePolicy.__greedy_outcome(qvs)
        greedy_actions = list()
        for v, a in np.vstack([qvs, actions]).T:
            if v == ou:
                if a in possible_actions:
                    greedy_actions.append(int(a))
        if len(greedy_actions) == 0:
            raise EvaluationExcpetion("No Q Values mapping to possible actions, cannot select greedy action")

        return greedy_actions[randint(0, len(greedy_actions)-1)]

    #
    # Save with class default filename.
    #
    def __save(self):
        self.save(self.__filename)

    #
    # Export the current policy to the given file name
    #
    def save(self, filename: str=None):
        if filename is not None and len(filename) > 0:
            self.__persistance.save(TemporalDifferencePolicy.__q_values,
                                    TemporalDifferencePolicy.__n,
                                    TemporalDifferencePolicy.__learning_rate_0,
                                    TemporalDifferencePolicy.__discount_factor,
                                    TemporalDifferencePolicy.__learning_rate_decay,
                                    filename)
        return

    #
    # Import the current policy to the given file name
    #
    def load(self, filename: str)-> Tuple[dict, int, np.float, np.float, np.float]:
        (TemporalDifferencePolicy.__q_values,
         TemporalDifferencePolicy.__n,
         TemporalDifferencePolicy.__learning_rate_0,
         TemporalDifferencePolicy.__discount_factor,
         TemporalDifferencePolicy.__learning_rate_decay) \
            = TemporalDifferencePolicy.__learning_rate_decay = self.__persistance.load(filename)

        return

    #
    # Q Values as a string (in grid form). This is just a visual debugger so it is
    # possible to see what q values are being selected from in the way that they
    # relate to the board. (3 x 3)
    #
    @classmethod
    def vals_and_actions_as_str(cls, q: [np.float], a: [int]) -> str:
        s = ""
        at = 0
        for i in range(0, 3):
            for j in range(0, 3):
                if a is not None and at < len(a) and a[at] == j+(i*3):
                    s += "[" + '{:+.16f}'.format(q[at]) + "] "
                    at += 1
                else:
                    s += "[                   ] "
            s += "\n"
        return s

