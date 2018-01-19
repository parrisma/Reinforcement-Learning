from State import State
from Environment import Environment


class TicTacToeState(State):

    #
    # Constructor has no arguments as it just sets the game
    #
    def __init__(self, env: Environment):
        self.__env = env

    #
    # An environment specific representation for Env. State
    #
    def state(self) -> object:
        return None

    #
    # An string representation of the environment state
    #
    def state_as_string(self) -> str:
        return None

    #
    # The environment the state was initiated with.
    #
    def get_env(self) -> Environment:
        return self.__env
