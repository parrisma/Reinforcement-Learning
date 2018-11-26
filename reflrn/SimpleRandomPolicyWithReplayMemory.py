import logging
from random import randint

from reflrn.DequeReplayMemory import DequeReplayMemory
from reflrn.Interface.Environment import Environment
from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


class SimpleRandomPolicyWithReplayMemory(Policy):

    #
    # At inti time the only thing needed is the universal set of possible
    # actions for the given Environment
    #
    def __init__(self,
                 lg: logging,
                 replay_memory: DequeReplayMemory,
                 env: Environment = None):
        self.__lg = lg
        self.__replay_memory = replay_memory
        self.__env = env
        self.__explain = False
        return

    #
    # Make a note of which environment policy is linked to.
    #
    def link_to_env(self, env: Environment) -> None:
        if self.__env is not None:
            raise Policy.PolicyAlreadyLinkedToEnvironment("Policy already linked to an environment !")
        self.__env = env
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

    #
    # Generate debug details when predicting actions.
    #
    @property
    def explain(self) -> bool:
        return self.__explain

    @explain.setter
    def explain(self, value: bool):
        if type(value) != bool:
            raise TypeError("explain property is type bool cannot not [" + type(value).__name__ + "]")
        self.__explain = value
