import abc

from reflrn.Interface.State import State


#
# This abstract base class is the fundamental contract with the Agents.
#


class Environment(metaclass=abc.ABCMeta):

    #
    # An array of all the environment attributes
    #
    @abc.abstractmethod
    def attribute_names(self) -> [str]:
        pass

    #
    # Get the named attribute
    #
    @abc.abstractmethod
    def attribute(self, attribute_name: str) -> object:
        pass

    #
    # Return an array of all the actions supported by this Environment.
    #
    # Key : action_id
    # Value : action info
    #
    @classmethod
    @abc.abstractmethod
    def actions(cls,
                state: State = None) -> [int]:
        pass

    #
    # True if the current episode in the environment has reached a terminal point. If a state is supplied
    # the function returns True if the given state represents a terminal state.
    #
    @abc.abstractmethod
    def episode_complete(self,
                         state: State = None) -> bool:
        pass

    #
    # Save the current Environment State
    #
    @abc.abstractmethod
    def save(self, file_name: str) -> None:
        pass

    #
    # Load the current Environment State
    #
    @abc.abstractmethod
    def load(self, file_name: str):
        pass

    #
    # Import the current Environment State from given State as string
    # ToDo: Remove and reconsider test strategy using test Agents
    #
    @abc.abstractmethod
    def import_state(self, state: str):
        pass

    #
    # Export the current Environment State to String
    # ToDo: Remove and reconsider test strategy using test Agents
    #
    @abc.abstractmethod
    def export_state(self) -> str:
        pass

    #
    # Run the given number of iterations
    #
    @abc.abstractmethod
    def run(self, iterations: int):
        pass

    #
    # Return the public environment attributes
    #
    @abc.abstractmethod
    def attributes(self) -> dict:
        pass

    #
    # Return the State of the environment
    #
    @abc.abstractmethod
    def state(self) -> State:
        pass
