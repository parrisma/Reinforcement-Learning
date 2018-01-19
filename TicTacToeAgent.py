from Agent import Agent
from State import State


class TicIacToeAgent(Agent):

    __env = None  # Environment to which agent is attached.

    def __init__(self, agent_id, agent_name):
        self.__id = agent_id
        self.__name = agent_name

    # Return immutable id
    #
    def id(self):
        return self.__id

    # Return immutable name
    #
    def name(self):
        return self.__name()

    #
    # Environment call back when environment shuts down
    #
    def terminate(self):
        return

    #
    # Environment call back when episode starts
    #
    def episode_init(self, state: State):
        return

    #
    # Environment call back when episode is completed
    #
    def episode_complete(self, state: State):
        return

    #
    # Environment call back to ask the agent to chose an action
    # given the environment in the current state.
    #
    def chose_action(self, state: State):
        return None

    #
    # Environment call back to reward agent for a play chosen for the given
    # state passed.
    #
    def reward(self, state: State, reward_for_play: float):
        return
