import abc
from reflrn.State import State

#
# For Debug, render given Q Values Dictionary as String.
#


class RenderQValsAsStr(metaclass=abc.ABCMeta):

    #
    # Render a given QVal Dictionary as a string.
    #
    @abc.abstractmethod
    def render_as_str(self, curr_state: State, q_vals: dict) -> str:
        pass
