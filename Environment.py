import abc

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
    # Return a dictionary of all the actions supported by this Environment.
    #
    # Key : action_id
    # Value : action info
    #
    @classmethod
    @abc.abstractmethod
    def actions(cls) -> [int]:
        pass

    #
    # True if the current episode in the environment has reached a terminal state.
    #
    # Environment specific summary of the terminal state of the environment.
    #
    @abc.abstractmethod
    def episode_complete(self) -> dict:
        pass

    #
    # Save the current Environment State
    #
    @abc.abstractmethod
    def save(self, file_name: str):
        pass

    #
    # Load the current Environment State
    #
    @abc.abstractmethod
    def load(self, file_name: str):
        pass

    #
    # Import the current Environment State from given State as string
    #
    @abc.abstractmethod
    def import_state(self, state: str):
        pass

    #
    # Export the current Environment State to String
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
