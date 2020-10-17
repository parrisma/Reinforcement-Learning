from reflrn.Interface.Agent import Agent
from reflrn.Interface.State import State


class TestAgent(Agent):
    actions = None
    action_idx = None

    def __init__(self,
                 agent_id: int,  # immutable & unique id for this agent
                 agent_name: str,
                 actions: list = None):
        self.__id = agent_id
        self.__name = agent_name
        if actions is not None:
            self.actions = actions.copy()
            self.action_idx = 0

    # Return immutable id
    #
    def id(self):
        return self.__id

    # Return immutable name
    #
    def name(self):
        return self.__name

    #
    # Roll back to the start of the defined action sequence
    #
    def __reset(self):
        if self.actions is not None:
            self.action_idx = 0

    #
    # Environment call back when environment shuts down
    #
    def terminate(self):
        self.__reset()
        return

    #
    # Environment call back when episode starts
    #
    def episode_init(self, state: State):
        self.__reset()
        return

    #
    # Environment call back when episode is complete
    #
    def episode_complete(self, state: State):
        self.__reset()
        return

    #
    # Environment call back to ask the agent to choose an action
    #
    # State : The current curr_coords of the environment
    # possible_actions : The set of possible actions the agent can play from this curr_coords
    #
    def choose_action(self, state: State, possible_actions: [int]) -> int:
        if self.actions is not None:
            if self.action_idx < len(self.actions):
                a = self.actions[self.action_idx]
                self.action_idx += 1
                return a
            else:
                raise TestAgent.ExceededNumberOfDefinedTestActions()
        return int(0)

    #
    #
    # Environment call back to reward agent for a play chosen for the given
    # curr_coords passed.
    #
    def reward(self, state: State, next_state: State, action: int, reward_for_play: float, episode_complete: bool):
        return

    # Called by the environment *once* at the start of the session
    # and the action set is given as dictionary
    #
    def session_init(self, actions: dict):
        self.__reset()
        return

    @property
    def explain(self) -> bool:
        raise NotImplementedError()

    # Exceeded number of defined test actions
    #
    class ExceededNumberOfDefinedTestActions(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)
