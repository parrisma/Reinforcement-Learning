from reflrn.Interface.Agent import Agent
from reflrn.Interface.State import State


class TestAgent(Agent):

    def __init__(self,
                 agent_id: int,  # immutable & unique id for this agent
                 agent_name: str):
        self.__id = agent_id
        self.__name = agent_name

    # Return immutable id
    #
    def id(self):
        return self.__id

    # Return immutable name
    #
    def name(self):
        return self.__name

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
    #
    # State : The current curr_coords of the environment
    # possible_actions : The set of possible actions the agent can play from this curr_coords
    #
    def chose_action(self, state: State, possible_actions: [int]) -> int:
        return int(0)

    #
    # Environment call back to reward agent for a play chosen for the given
    # curr_coords passed.
    #
    def reward(self, state: State, next_state: State, action: int, reward_for_play: float, episode_complete: bool):
        return

    #
    # Called by the environment *once* at the start of the session
    # and the action set is given as dictionary
    #
    def session_init(self, actions: dict):
        return
