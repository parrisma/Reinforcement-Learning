import abc

#
# This abstract makes a play based just on possible moves given. This is called
# when the e-greedy asks for a random play to explore the state space. This can
# be pure random or informed random to try an expose more significant areas of
# state space. e.g. with some manually coded strategy for the given environment.
#


class ExplorationPlay(metaclass=abc.ABCMeta):

    #
    # Select an action from the possible actions supplied.
    #
    @abc.abstractmethod
    def select_action(self, possible_actions: [int]) -> int:
        pass
