import abc


#
# This is the interface that all Grid World Grids need to implement.
#


class Grid(metaclass=abc.ABCMeta):

    #
    # Immutable Id for the Grid
    #
    @abc.abstractmethod
    def id(self) -> int:
        pass

    #
    # Return the list of possible actions
    #
    @abc.abstractmethod
    def actions(self) -> [int]:
        pass

    #
    # Execute the given (allowable) action and return the reward and the state
    #
    @abc.abstractmethod
    def execute_action(self, action: int) -> int:
        pass

    #
    # Deep Copy the Grid.
    #
    @abc.abstractmethod
    def copy(self):
        pass

    #
    # What actions are allowable with agent at current location.
    #
    @abc.abstractmethod
    def allowable_actions(self) -> [int]:
        pass
