import logging
from random import randint

import numpy as np

from reflrn import Agent
from reflrn import Environment
from reflrn import State
from .Grid import Grid
from .GridWorldState import GridWorldState


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
            state = GridWorldState(self.__board, self.__x_agent, self.__o_agent)
            self.__x_agent.episode_init(state)
            self.__o_agent.episode_init(state)
            agent = (self.__x_agent, self.__o_agent)[randint(0, 1)]

            while not self.episode_complete():
                state = GridWorldState(self.__board, self.__x_agent, self.__o_agent)
                self.__lg.debug(agent.name())
                self.__lg.debug(state.state_as_string())
                self.__lg.debug(state.state_as_visualisation())
                agent = self.__play_action(agent)
                i += 1
                if i % 500 == 0:
                    self.__lg.info("Iteration: " + str(i))

            self.__keep_stats()
            self.__lg.debug("Episode Complete")
            state = GridWorldState(self.__board, self.__x_agent, self.__o_agent)
            self.__lg.debug(state.state_as_visualisation())
            self.__x_agent.episode_complete(state)
            self.__o_agent.episode_complete(state)
        self.__x_agent.terminate()
        self.__o_agent.terminate()
        return

    @classmethod
    def no_agent(cls):
        return cls.__no_agent

    #
    # Return the actions as a list of integers.
    #
    @classmethod
    def actions(cls) -> [int]:
        return list(map(lambda a: int(a), list(GridWorld.__actions.keys())))

    #
    # Make the play chosen by the given agent. If it is a valid play
    # confer reward and switch play to other agent. If invalid play
    # i.e. play in a cell where there is already a marker confer
    # penalty and leave play with the same agent.
    # ToDo
    def __play_action(self, agent: Agent) -> Agent:

        other_agent = self.__next_agent[agent.name()]
        state = GridWorldState(self.__board, self.__x_agent, self.__o_agent)

        # Make the play on the board.
        action = agent.chose_action(state, self.__actions_ids_left_to_take())
        if action not in self.__actions_ids_left_to_take():
            print("Opps")
        self.__take_action(self.__actions[action], agent)
        next_state = GridWorldState(self.__board, self.__x_agent, self.__o_agent)

        if self.episode_complete():
            attributes = self.attributes()
            if attributes[self.attribute_won[0]]:
                agent.reward(state, next_state, action, self.__win, True)
                return None  # episode complete - no next agent to go
            if attributes[self.attribute_draw[0]]:
                agent.reward(state, next_state, action, self.__draw, True)
                return None  # episode complete - no next agent to go

        agent.reward(state, next_state, action, self.__play, False)
        return other_agent  # play moves to next agent

    #
    # The episode is over if one agent has made a line of three on
    # any horizontal, vertical or diagonal or if there are no actions
    # left to take and neither agent has won.
    #
    def episode_complete(self):
        return self.__grid.episode_complete()

    #
    # Convert an environment (board) from a string form to the
    # internal board state.
    #
    def __string_to_internal_state(self, moves_as_str):
        mvs = moves_as_str.split('~')
        if moves_as_str is not None:
            for mv in mvs:
                if len(mv) > 0:
                    pl, ps = mv.split(":")
                    self.__take_action(int(ps), int(pl))
        return

    #
    # Convert internal (board) state to string
    #
    def __internal_state_to_string(self, board) -> str:
        mvs = ""
        bd = np.reshape(self.__board, self.__board.size)
        cell_num = 0
        for actor in bd:
            if not np.isnan(actor):
                mvs += str(int(actor)) + ":" + str(int(cell_num + 1)) + "~"
            cell_num += 1
        if len(mvs) > 0:
            mvs = mvs[:-1]
        return mvs

    #
    # The current state of the environment as string
    #
    def state_as_str(self) -> str:
        return self.__internal_state_to_string(self.__board)

    #
    # Load Environment from file
    #
    def load(self, file_name: str):
        raise NotImplementedError("load() not implemented for TicTacTo")

    #
    # Save Environment to file
    #
    def save(self, file_name: str):
        raise NotImplementedError("save() not implemented for TicTacTo")

    #
    # Expose current environment state as string
    #
    def export_state(self):
        return self.__internal_state_to_string(self.__board)

    #
    # Set environment state from string
    #
    def import_state(self, state_as_string):
        self.__string_to_internal_state(state_as_string)

    #
    # Return the State of the environment
    #
    def state(self) -> State:
        return GridWorldState(self.__board)
