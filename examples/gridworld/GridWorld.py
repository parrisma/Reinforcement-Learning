import logging

import numpy as np

from reflrn.Interface.Agent import Agent
from reflrn.Interface.Environment import Environment
from reflrn.Interface.State import State
from .Grid import Grid
from .GridWorldState import GridWorldState
from .IllegalGridMoveException import IllegalGridMoveException


class GridWorld(Environment):

    #
    # Boot strap from a single agent and single grid
    #
    def __init__(self, x: Agent, grid: Grid, lg: logging):
        self.__lg = lg
        self.__grid = grid
        self.__x_agent = x
        return

    #
    # Return game to initial state.
    #
    def reset(self):
        self.__grid.reset()
        return

    #
    # An array of all the environment attributes
    #
    def attribute_names(self) -> [str]:
        return None

    #
    # Get the named attribute
    #
    def attribute(self, attribute_name: str) -> object:
        return None

    #
    # Run the given number of iterations
    #
    def run(self, iterations: int):
        i = 0
        self.__keep_stats(reset=True)
        while i <= iterations:
            self.__lg.debug("Start Episode")
            self.reset()
            state = GridWorldState(self.__grid)
            self.__x_agent.episode_init(state)

            agent = self.__x_agent
            while not self.episode_complete():
                state = GridWorldState(self.__grid)
                self.__lg.debug(agent.name())
                self.__lg.debug(state.state_as_string())
                self.__lg.debug(state.state_as_visualisation())
                agent = self.__play_action(agent)
                i += 1
                if i % 500 == 0:
                    self.__lg.info("Iteration: " + str(i))

            self.__keep_stats()
            self.__lg.debug("Episode Complete")
            state = GridWorldState(self.__grid)
            self.__lg.debug(state.state_as_visualisation())
            self.__x_agent.episode_complete(state)
        self.__x_agent.terminate()
        return

    #
    # record statistics for episode.
    #
    def __keep_stats(self, reset: bool = False):
        pass

    @classmethod
    def no_agent(cls):
        return None

    #
    # Return the actions as a list of integers.
    #
    def actions(self) -> [int]:
        return self.__grid.actions()

    #
    # Make the play chosen by the given agent. If it is a valid play
    # confer reward and switch play to other agent. If invalid play
    # i.e. play in a cell where there is already a marker confer
    # penalty and leave play with the same agent.
    # ToDo
    def __play_action(self, agent: Agent) -> Agent:

        state = GridWorldState(self.__grid)

        # Make the play on the board.
        action = agent.chose_action(state, self.__grid.allowable_actions())
        if action not in self.__grid.allowable_actions():
            raise IllegalGridMoveException("Action chosen by agent is not allowable from current location on grid ["
                                           + str(action) + "]")
        reward = self.__take_action(action, agent)
        next_state = GridWorldState(self.__grid)

        if self.episode_complete():
            agent.reward(state, next_state, action, reward, True)
            return None  # episode complete - no next agent to go
        else:
            agent.reward(state, next_state, action, reward, False)
            return agent  # play stays with (only) agent

    #
    #
    #
    def __take_action(self, action: int, agent: Agent) -> np.float:
        return self.__grid.execute_action(action)

    #
    # The episode is over if one agent has made a line of three on
    # any horizontal, vertical or diagonal or if there are no actions
    # left to take and neither agent has won.
    #
    def episode_complete(self):
        return self.__grid.episode_complete()

    #
    # The current state of the environment as string
    #
    def state_as_str(self) -> str:
        return GridWorldState(self.__grid, self.__x_agent).state_as_string()

    #
    # Load Environment from file
    #
    def load(self, file_name: str):
        raise NotImplementedError("load() not implemented for GridWorld")

    #
    # Save Environment to file
    #
    def save(self, file_name: str):
        raise NotImplementedError("save() not implemented for GridWorld")

    #
    # Expose current environment state as string
    #
    def export_state(self):
        raise NotImplementedError("export_state() not implemented for GridWorld")

    #
    # Set environment state from string
    #
    def import_state(self, state_as_string):
        raise NotImplementedError("import_state() not implemented for GridWorld")

    #
    # Return the State of the environment
    #
    def state(self) -> State:
        return GridWorldState(self.__grid, self.__x_agent)

    #
    # No attributes supported at this point
    #
    def attributes(self) -> dict:
        return None
