from Agent import Agent
from State import State
from random import randint


class RandomNonLearningAgent(Agent):

    __actions = None
    __total_reward = float(0)

    def __init__(self, agent_id: int, agent_name: str):
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
        self.__total_reward = float(0)
        return

    #
    # Environment call back when episode is completed
    #
    def episode_complete(self, state: State):
        print(self.__name + " final reward : " + str(self.__total_reward))
        return

    #
    # Environment call back to ask the agent to chose an action
    # given the environment in the current state.
    #
    def chose_action(self, state: State):
        action = self.__actions[randint(0, len(self.__actions)-1)]
        print(self.__name + " chose action : " + str(action))
        return action

    #
    # Environment call back to reward agent for a play chosen for the given
    # state passed.
    #
    def reward(self, state: State, reward_for_play: float):
        print(self.__name + " reward: " + str(reward_for_play))
        self.__total_reward += reward_for_play
        return

    #
    # Called by the environment *once* at the start of the session
    # and the action set is given as dictionary
    #
    def session_init(self, actions: dict):
        self.__actions = actions
