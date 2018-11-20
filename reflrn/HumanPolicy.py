import logging

from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State
from reflrn.Interface.Environment import Environment


class HumanPolicy(Policy):

    def link_to_env(self, env: Environment) -> None:
        return

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self, agent_name: str, lg: logging):
        self.__lg = lg
        self.__agent_name = agent_name
        return

    #
    # No policy update, just show the resulting curr_coords/reward.
    #
    def update_policy(self,
                      agent_name: str,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool):
        return

    #
    # Greedy action; request human user to input action.
    #
    def select_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:
        print("-")
        print(state.state_as_array())
        print("-")
        input_request = "Make you move selection as agent: " + self.__agent_name + " from possible actions ("
        for mv in possible_actions:
            input_request += str(mv + 1) + ", "
        input_request += "): "

        mv = -1
        while mv not in possible_actions:
            mv = int(input(input_request)) - 1
        return mv

    #
    # Export the current policy to the given file name
    #
    def save(self, filename: str = None):
        self.__lg.warning("Save not supported for Human Policy")
        return

    #
    # Import the current policy to the given file name
    #
    def load(self, filename: str = None):
        self.__lg.warning("Load not supported for Human Policy")
        return
