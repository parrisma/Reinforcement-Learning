import logging
from random import randint

from reflrn.DequeReplayMemory import DequeReplayMemory
from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


class SimpleRandomPolicyWithReplayMemory(Policy):

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self,
                 lg: logging,
                 replay_memory: DequeReplayMemory):
        self.__lg = lg
        self.__replay_memory = replay_memory
        return

    #
    # Policy is always totally random, so no internal policy curr_coords is needed
    #
    # prev_state : the previous curr_coords for this Agent; None if no previous curr_coords
    # prev_action : the previous action of this agent; has no meaning is prev_state = None
    # curr_coords : current curr_coords of the environment *after* the given action was played
    # action : the action played by this agent that moved the curr_coords to the curr_coords passed
    # reward : the reward associated with the given curr_coords/action pair.
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
        self.__replay_memory.append_memory(state, next_state, action, reward, episode_complete)
        return

    #
    # Greedy action; this is random policy so just return any random action from the
    # set of actions.
    #
    def select_action(self, agent_name: str, state: State, possible_actions) -> int:
        return possible_actions[randint(0, len(possible_actions) - 1)]

    def save(self, filename: str = None):
        return

    def load(self, filename: str = None):
        return
