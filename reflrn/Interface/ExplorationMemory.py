import abc

from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


#
# Keep track of exploration events.
#


class ExplorationMemory(metaclass=abc.ABCMeta):
    class Memory:
        EPISODE = int(1)
        POLICY = int(2)
        AGENT = int(3)
        STATE = int(4)
        ACTION = int(5)

    LAST_EPISODE = int(-1)
    ALL_EPISODES = int(-2)

    #
    # Add an exploration memory.
    #
    @abc.abstractmethod
    def add(self,
            episode_number: int,
            policy: Policy,
            agent_name: str,
            state: State,
            next_state: State,
            action: int,
            reward: float,
            episode_complete: bool):
        pass

    #
    # Get Memories
    #
    @abc.abstractmethod
    def get(self,
            get_by: Memory,
            episode: int):
        pass
