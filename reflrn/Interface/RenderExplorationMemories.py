import abc

from reflrn.Interface.ExplorationMemory import ExplorationMemory


#
# Render given exploration memory in visual way for debugging.
#


class RenderExplorationMemory(metaclass=abc.ABCMeta):

    #
    # Render a given exploration memory episode
    #
    @abc.abstractmethod
    def render_episode(self, exploration_memory: ExplorationMemory, episode: int) -> str:
        pass
