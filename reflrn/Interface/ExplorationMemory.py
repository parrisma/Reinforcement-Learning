import abc

from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


#
# Keep track of exploration events.
#


class ExplorationMemory(metaclass=abc.ABCMeta):
    class Memory:
        EPISODE = int(0)
        POLICY = int(1)
        AGENT = int(2)
        STATE = int(3)
        NEXT_STATE = int(4)
        ACTION = int(5)
        REWARD = int(6)
        EPISODE_COMPLETE = int(7)
        MEM_TYPES = (EPISODE, POLICY, AGENT, STATE, NEXT_STATE, ACTION, REWARD, EPISODE_COMPLETE)

    class ExplorationMemoryException(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    class ExplorationMemoryMemTypeSearchNotSupported(ExplorationMemoryException):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    class ExplorationMemoryNoSuchEpisode(ExplorationMemoryException):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    SUPPORTED_GETBY_INDEX = (Memory.POLICY,
                             Memory.AGENT,
                             Memory.STATE,
                             Memory.ACTION)

    UNSUPPORTED_GETBY_INDEX = (Memory.EPISODE,
                               Memory.NEXT_STATE,
                               Memory.REWARD,
                               Memory.EPISODE_COMPLETE)

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
            episode_complete: bool) -> None:
        pass

    #
    # Get Memories by Type.
    #
    # Use Case : I want all of the memories for the given type either in the last episode in which that type
    # appeared or across all episodes in which that type appeared.
    #
    @abc.abstractmethod
    def get_memories_by_type(self,
                             get_by: Memory,
                             value: object
                             ) -> [[], [], [], [], [], [], [], []]:
        pass

    #
    # Get Memories by Episode
    #
    # Use Case : I want all of the memories from the given episode.
    #
    @abc.abstractmethod
    def get_memories_by_episode(self,
                                episode: int
                                ) -> [[], [], [], [], [], [], [], []]:
        pass
