import abc

#
# This abstract base class is the fundamental contract with the Agents.
#


class Environment(metaclass=abc.ABCMeta):

    #
    # Return a dictionary of all the actions supported by this Environment.
    #
    # Key : action_id
    # Value : action info
    #
    @classmethod
    @abc.abstractmethod
    def actions(cls) -> dict:
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
    def save(self, file_name):
        pass

    #
    # Load the current Environment State
    #
    @abc.abstractmethod
    def load(self, file_name):
        pass

    #
    # Import the current Environment State from given State as string
    #
    @abc.abstractmethod
    def import_state(self, environment_as_string):
        pass

    #
    # Export the current Environment State to String
    #
    @abc.abstractmethod
    def export_state(self) -> str:
        pass
