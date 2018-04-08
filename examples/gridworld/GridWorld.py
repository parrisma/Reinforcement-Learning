import numpy as np
import logging
from random import randint
from reflrn import Environment
from reflrn import Agent
from reflrn import State
from .GridWorldState import GridWorldState


class GridWorld(Environment):

    # The "game board" is passed in at construction time.

    __actions = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}  # N, S ,E, W (row-offset, col-offset)
    asStr = True
    __episode = 'episode number'

    #
    # Constructor has no arguments as it just sets the game
    # to an initial up-played set-up
    #
    def __init__(self, x: Agent, lg: logging):
        self.__lg = lg
        self.__board = GridWorld.__empty_board()
        self.__last_board = None
        self.__agent = GridWorld.__no_agent
        self.__last_agent = GridWorld.__no_agent
        self.__x_agent = x
        self.__x_agent.session_init(self.actions())
        self.__stats = None
        return

    #
    # Return game to initial state, where no one has played
    # and the board contains no moves.
    #
    def reset(self):
        self.__board = GridWorld.__empty_board()
        self.__last_board = None
        self.__agent = GridWorld.__no_agent
        self.__last_agent = GridWorld.__no_agent
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
    # Keep stats of wins by agent.
    #
    def __keep_stats(self, reset: bool=False):
        if reset is True or self.__stats is None:
            self.__stats = dict()
            self.__stats[self.__episode] = 1
            self.__stats[self.__o_agent.name()] = 0
            self.__stats[self.__x_agent.name()] = 0

        if self.__episode_won():
            self.__stats[self.__agent.name()] = self.__stats[self.__agent.name()] + 1
            self.__stats[self.__episode] = self.__stats[self.__episode] + 1
            self.__lg.debug(self.__agent.name() + " Wins")

        if self.__stats[self.__episode] % 100 == 0:
            self.__lg.info("Stats: Agent : " + self.__x_agent.name() + " [" +
                           str(round((self.__stats[self.__x_agent.name()] / self.__stats[self.__episode]) * 100)) + "%] " +
                           "Agent : " + self.__o_agent.name() + " [" +
                           str(round((self.__stats[self.__o_agent.name()] / self.__stats[self.__episode]) * 100)) + "%] "
                          )
        return


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

    #
    # Return a new empty board.
    #
    @classmethod
    def __empty_board(cls):
        return np.full((3, 3), np.nan)

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
    # Assume the play_action has been validated by play_action method
    # Make a copy of board before play_action is made and the last player
    #
    def __take_action(self, action: int, agent: Agent):
        self.__last_board = np.copy(self.__board)
        self.__last_agent = self.__agent
        self.__agent = agent
        self.__board[action] = self.__agent.id()
        return

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
    # Return the attributes of the environment
    #
    def attributes(self):
        attr_dict = dict()
        attr_dict[GridWorld.attribute_draw[0]] = not self.__actions_left_to_take()
        attr_dict[GridWorld.attribute_won[0]] = self.__episode_won()
        attr_dict[GridWorld.attribute_complete[0]] = \
            attr_dict[GridWorld.attribute_draw[0]] or attr_dict[GridWorld.attribute_won[0]]
        attr_dict[GridWorld.attribute_agent[0]] = self.__agent
        attr_dict[GridWorld.attribute_board[0]] = np.copy(self.__board)
        return attr_dict

    #
    # Is there a winning move on the board.
    #
    def __episode_won(self):
        rows = np.abs(np.sum(self.__board, axis=1))
        cols = np.abs(np.sum(self.__board, axis=0))
        diag_lr = np.abs(np.sum(self.__board.diagonal()))
        diag_rl = np.abs(np.sum(np.rot90(self.__board).diagonal()))

        if np.sum(rows == 3) > 0:
            return True
        if np.sum(cols == 3) > 0:
            return True
        if not np.isnan(diag_lr):
            if ((np.mod(diag_lr, 3)) == 0) and diag_lr > 0:
                return True
        if not np.isnan(diag_rl):
            if ((np.mod(diag_rl, 3)) == 0) and diag_rl > 0:
                return True
        return False

    #
    # Are there any remaining actions to be taken >
    #
    def __actions_left_to_take(self):
        return self.__board[np.isnan(self.__board)].size > 0

    #
    # Are there any remaining actions to be taken >
    #
    def __actions_ids_left_to_take(self):
        alt = np.reshape(self.__board, self.__board.size)
        alt = np.asarray(self.actions())[np.isnan(alt) == True]
        return alt

    #
    # The episode is over if one agent has made a line of three on
    # any horizontal, vertical or diagonal or if there are no actions
    # left to take and neither agent has won.
    #
    def episode_complete(self):
        if self.__episode_won() or not self.__actions_left_to_take():
            return True
        return False

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
                mvs += str(int(actor)) + ":" + str(int(cell_num+1))+"~"
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
