import numpy as np
from Policy import Policy
from State import State
from TemporalDifferencePolicyPersistance import TemporalDifferencePolicyPersistance
from EvaluationException import EvaluationExcpetion
from random import randint
from FixedGames import FixedGames
from typing import Tuple


class HumanPolicy(Policy):

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self, agent_name: str):
        self.__agent_name = agent_name
        return

    #
    # No policy update, just show the resulting state/reward.
    #
    def update_policy(self, agent_name: str, prev_state: State, prev_action: int, state: State, action: int, reward: float):
        return

    #
    # Greedy action; request human user to input action.
    #
    def greedy_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:
        input_request = "Make you move selection as agent: " + self.__agent_name + " from possible actions ("
        for mv in possible_actions:
            input_request += str(mv+1)+", "
        input_request += "): "

        mv = -1
        while mv not in possible_actions:
            mv = int(input(input_request))-1
        return mv

    #
    # Export the current policy to the given file name
    #
    def save(self, filename: str=None):
        raise NotImplementedError("Save not supported for Human Policy")

    #
    # Import the current policy to the given file name
    #
    def load(self, filename: str)-> Tuple[dict, int, np.float, np.float, np.float]:
        raise NotImplementedError("Load not supported for Human Policy")
