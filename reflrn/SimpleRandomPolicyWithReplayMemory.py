import logging
from random import randint

from reflrn.Policy import Policy
from reflrn.ReplayMemory import ReplayMemory
from reflrn.State import State


class SimpleRandomPolicyWithReplayMemory(Policy):

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self,
                 lg: logging,
                 replay_memory: ReplayMemory):
        self.__lg = lg
        self.__replay_memory = replay_memory
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
    # We don't learn from the reply memory but we must record the memories so the other Actor/Critic
    # agents have the whole set of memories to train from.
    #
    def update_policy(self,
                      agent_name: str,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool):
        self.__replay_memory.appendMemory(state, next_state, action, reward, episode_complete)
        return

    #
    # Greedy action; this is random policy so just return any random action from the
    # set of actions.
    #
    def greedy_action(self, agent_name: str, state: State, possible_actions) -> int:
        return possible_actions[randint(0, len(possible_actions) - 1)]

    def save(self, filename: str = None):
        return

    def load(self, filename: str = None):
        return
