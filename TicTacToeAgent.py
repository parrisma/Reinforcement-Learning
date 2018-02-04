import random
import logging
from EvaluationException import EvaluationExcpetion
from State import State
from random import randint
from Agent import Agent
from Policy import Policy
from ExplorationPlay import ExplorationPlay


class TicTacToeAgent(Agent):

    def __init__(self,
                 agent_id: int,           # immutable & unique id for this agent
                 agent_name: str,         # immutable & unique name for this agent
                 policy: Policy,          # the policy to drive action selection.
                 epsilon_greedy: float,   # if random() > epsilon greedy take greedy action else random action
                 exploration_play: ExplorationPlay,  # Play to take when exploring (i.e. when not being greedy)
                 lg: logging):
        self.__lg = lg
        self.__id = agent_id
        self.__name = agent_name
        self.__policy = policy
        self.__epsilon_greedy = epsilon_greedy
        self.__exploration = exploration_play

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
        self.__policy.save()
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
    # State : The current state of the environment
    # possible_actions : The set of possible actions the agent can play from this state
    #
    def chose_action(self, state: State, possible_actions: [int]) -> int:

        # if random() > epsilon greedy then take greedy action else a random action
        if random.random() > self.__epsilon_greedy:
            try:
                # If there are q values for given state we can predict a greedy action
                action = self.__policy.greedy_action(self.__name, state, possible_actions)
                self.__lg.debug(self.__name + " chose greedy action : " + str(action+1))
            except EvaluationExcpetion:
                # cannot predict a greedy action so random
                action = self.__exploration.select_action(possible_actions)
                self.__lg.debug(self.__name + " chose exploration action : " + str(action+1))
        else:
            action = self.__exploration.select_action(possible_actions)
            self.__lg.debug(self.__name + " chose exploration action : " + str(action+1))

        return action

    #
    # Environment call back to reward agent for a play chosen for the given
    # state passed.
    #
    def reward(self, state: State, next_state: State, action: int, reward_for_play: float, episode_complete: bool):
        self.__policy.update_policy(self.name(),
                                    state,
                                    next_state,
                                    action,
                                    reward_for_play,
                                    episode_complete)
        return

    #
    # Called by the environment *once* at the start of the session
    # and the action set is given as dictionary
    #
    def session_init(self, actions: dict):
        return
