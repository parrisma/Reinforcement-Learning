import abc

from reflrn.Interface.State import State


#
# Reply Memory for holding keeping learning events to support real time (dynamic) model training.
#


class ReplayMemory(metaclass=abc.ABCMeta):

    #
    # Add a memory to the reply memory, but tag it with the episode id such that whole episodes
    # can later be recovered for training.
    #
    @abc.abstractmethod
    def append_memory(self,
                      state: State,
                      next_state: State,
                      action: int,
                      reward: float,
                      episode_complete: bool) -> None:
        pass

    #
    # How many items in the replay memory
    #
    @abc.abstractmethod
    def len(self) -> int:
        pass

    #
    # Get a random set of memories
    #
    @abc.abstractmethod
    def get_random_memories(self, sample_size: int) -> [[[], [], [], [], [], []]]:
        pass

    #
    # Get just the last memory with respect to the given state. If given state is
    # None return the last memory overall.
    #
    @abc.abstractmethod
    def get_last_memory(self, state: State = None) -> [[], [], [], [], [], []]:
        pass
