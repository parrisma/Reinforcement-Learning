import abc

#
# This is an interface specification for an reinforcement learning agent
#


class Agent(metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractclassmethod
    def learn(cls, state, action, reward):
        pass

    @classmethod
    @abc.abstractclassmethod
    def act(cls, state):
        pass
