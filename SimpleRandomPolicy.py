from Policy import Policy
from State import State
from random import randint


class SimpleRandomPolicy(Policy):

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self):
        return

    #
    # Policy is always totally random, so no internal policy state is needed
    #
    # prev_state : the previous state for this Agent; None if no previous state
    # prev_action : the previous action of this agent; has no meaning is prev_state = None
    # state : current state of the environment *after* the given action was played
    # action : the action played by this agent that moved the state to the state passed
    # reward : the reward associated with the given state/action pair.
    #
    def update_policy(self, agent_name: str, prev_state: State, prev_action: int, state: State, action: int, reward: float):
        return

    #
    # Greedy action; this is random policy so just return any random action from the
    # set of actions.
    #
    def greedy_action(self, agent_name: str, state: State, possible_actions) -> int:
        return possible_actions[randint(0, len(possible_actions)-1)]
