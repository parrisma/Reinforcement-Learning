from reflrn import ExplorationPlay
from random import randint


class PureRandomExploration(ExplorationPlay):

    #
    # This is a pure random play, just pick any of the possible actions.
    #
    def select_action(self, possible_actions: [int]) -> int:
        return possible_actions[randint(0, len(possible_actions) - 1)]