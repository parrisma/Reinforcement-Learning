import logging
from random import randint

import numpy as np

from examples.tictactoe.TicTacToeState import TicTacToeState
from reflrn.Interface.Agent import Agent
from reflrn.Interface.Environment import Environment
from reflrn.Interface.State import State


class TicTacToe(Environment):
    # There are 5812 legal board states that can be reached before there is a winner
    # http://brianshourd.com/posts/2012-11-06-tilt-number-of-tic-tac-toe-boards.html

    __play = float(-1)  # reward for playing an action
    __draw = float(-10)  # reward for playing to end but no one wins
    __win = float(100)  # reward for winning a game
    __no_agent = None
    __win_mask = np.full((1, 3), 3, np.int8)
    __actions = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}
    __drawn = "draw"
    __games = "games"
    __states = "states"
    empty_cell = np.nan  # value of a free action space on board
    asStr = True
    attribute_draw = ("Draw", "True if episode ended in a drawn curr_coords", bool)
    attribute_won = ("Won", "True if episode ended in a win curr_coords", bool)
    attribute_complete = ("Complete", "True if the environment is in a complete curr_coords for any reason", bool)
    attribute_agent = ("agent", "The last agent to make a move", Agent)
    attribute_board = (
        "board", "The game board as a numpy array (3,3), np.nan => no move else the id of the agent", np.array)
    __episode = 'episode number'
    __random_turns = True

    #
    # Constructor has no arguments as it just sets the game
    # to an initial up-played set-up
    #
    def __init__(self, x: Agent,
                 o: Agent,
                 lg: logging,
                 save_on_exit: bool = False):
        self.__lg = lg
        self.__board = TicTacToe.__empty_board()
        self.__last_board = None
        self.__agent = TicTacToe.__no_agent
        self.__last_agent = TicTacToe.__no_agent
        self.__x_agent = x
        self.__o_agent = o
        self.__next_agent = {x.name(): o, o.name(): x}
        self.__x_agent.session_init(self.actions())
        self.__o_agent.session_init(self.actions())
        self.__agents = dict()
        self.__agents[self.__o_agent.id()] = self.__o_agent
        self.__agents[self.__x_agent.id()] = self.__x_agent
        self.__stats = None
        self.__save_on_exit = save_on_exit
        return

    #
    # Return game to initial curr_coords, where no one has played
    # and the board contains no moves.
    #
    def reset(self):
        self.__board = TicTacToe.__empty_board()
        self.__last_board = None
        self.__agent = TicTacToe.__no_agent
        self.__last_agent = TicTacToe.__no_agent
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
    # Reset the statistics.
    #
    def __reset_stats(self) -> None:
        self.__stats = dict()
        self.__stats[self.__episode] = 1
        self.__stats[self.__o_agent.name()] = 0
        self.__stats[self.__x_agent.name()] = 0
        self.__stats[self.__drawn] = 0
        self.__stats[self.__games] = dict()
        self.__stats[self.__states] = dict()
        return

    #
    # Keep count for given dict attr
    #
    def __keep_count(self,
                     attr: str,
                     key_as_str: str) -> None:
        if key_as_str not in self.__stats[attr]:
            (self.__stats[attr])[key_as_str] = 0
        (self.__stats[attr])[key_as_str] += 1

    #
    # Keep stats for each step
    #
    def __keep_step_stats(self,
                          state: State) -> None:
        self.__keep_count(attr=self.__states, key_as_str=state.state_as_string())
        return

    #
    # Keep stats at end of episode
    #
    def __keep_episode_stats(self,
                             state: State) -> None:
        episode_summary = self.attributes()
        self.__keep_count(attr=self.__games, key_as_str=state.state_as_string())
        if episode_summary[TicTacToe.attribute_won[0]]:
            agnt = episode_summary[TicTacToe.attribute_agent[0]].name()
            self.__stats[agnt] += 1
            self.__stats[self.__episode] = self.__stats[self.__episode] + 1
            self.__lg.debug(agnt + "Wins")

        if episode_summary[TicTacToe.attribute_draw[0]]:
            self.__stats[self.__drawn] += 1
            self.__stats[self.__episode] = self.__stats[self.__episode] + 1
            self.__lg.debug("Episode Drawn")

        if self.__stats[self.__episode] % 100 == 0:
            self.__lg.info("Stats: Agent : " + self.__x_agent.name() + " [" +
                           str(round(
                               (self.__stats[self.__x_agent.name()] / self.__stats[self.__episode]) * 100)) + "%] " +
                           "Agent : " + self.__o_agent.name() + " [" +
                           str(round(
                               (self.__stats[self.__o_agent.name()] / self.__stats[self.__episode]) * 100)) + "%] " +
                           "Draw : [" +
                           str(round(
                               (self.__stats[self.__drawn] / self.__stats[self.__episode]) * 100)) + "%] "
                           )
        return

    #
    # Run the given number of iterations
    #
    def run(self, iterations: int):
        i = 0
        self.__reset_stats()
        while i <= iterations:
            self.__lg.debug("Start Episode")
            self.reset()
            state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)
            self.__x_agent.episode_init(state)
            self.__o_agent.episode_init(state)

            if self.__random_turns:
                agent = (self.__x_agent, self.__o_agent)[randint(0, 1)]
            else:
                agent = self.__x_agent

            while not self.episode_complete():
                state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)
                self.__lg.debug(agent.name())
                self.__lg.debug(state.state_as_string())
                self.__lg.debug(state.state_as_visualisation())
                agent = self.__play_action(agent)
                self.__keep_step_stats(state)
                i += 1
                if i % 500 == 0:
                    self.__lg.info("Iteration: " + str(i))

            self.__keep_episode_stats(state)
            self.__lg.debug("Episode Complete")
            state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)
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
    # Return the actions as a list of integers. If not state is given return the list of all
    # action else return the list of actions valid in this state.
    #
    @classmethod
    def actions(cls,
                state: State = None) -> dict:

        if state is None:
            return TicTacToe.__actions
        else:
            return np.array(list(TicTacToe.__actions.keys()))[np.isnan(state.state()).reshape(9)]

    #
    # Assume the play_action has been validated by play_action method
    # Make a deep_copy of board before play_action is made and the last player
    #
    def __take_action(self, action: int, agent: Agent):
        self.__last_board = np.copy(self.__board)
        self.__last_agent = self.__agent
        self.__agent = agent
        self.__board[self.__actions[action]] = self.__agent.id()
        return

    #
    # Make the play chosen by the given agent. If it is a valid play
    # confer reward and switch play to other agent. If invalid play
    # i.e. play in a cell where there is already a marker confer
    # penalty and leave play with the same agent.
    # ToDo
    def __play_action(self, agent: Agent) -> Agent:

        other_agent = self.__next_agent[agent.name()]
        state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)

        # Make the play on the board.
        self.__lg.debug(state.state_as_array())
        action = agent.choose_action(state, self.__actions_ids_left_to_take())
        if action not in self.__actions_ids_left_to_take():
            raise TicTacToe.IllegalActorAction("Actor Proposed Illegal action in current state :" + str(action))
        self.__take_action(action, agent)
        next_state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)

        if self.episode_complete():
            attributes = self.attributes()
            if attributes[self.attribute_won[0]]:
                agent.reward(state, next_state, action, self.__win, True)
                other_agent.reward(state, next_state, action, self.__win, False)  # block not win
                si = state.invert_player_perspective()
                nsi = next_state.invert_player_perspective()
                agent.reward(si, nsi, action, self.__win, False)  # block not win
                other_agent.reward(si, nsi, action, self.__win, True)
                return None  # episode complete - no next agent to go
            if attributes[self.attribute_draw[0]]:
                agent.reward(state, next_state, action, self.__draw, True)
                other_agent.reward(state, next_state, action, self.__draw, True)
                return None  # episode complete - no next agent to go

        agent.reward(state, next_state, action, self.__play, False)
        return other_agent  # play moves to next agent

    #
    # Return the attributes of the environment
    #
    def attributes(self):
        attr_dict = dict()
        attr_dict[TicTacToe.attribute_won[0]] = self.__episode_won()
        attr_dict[TicTacToe.attribute_draw[0]] = False
        if not attr_dict[TicTacToe.attribute_won[0]]:
            attr_dict[TicTacToe.attribute_draw[0]] = not self.__actions_left_to_take()
        attr_dict[TicTacToe.attribute_complete[0]] = \
            attr_dict[TicTacToe.attribute_draw[0]] or attr_dict[TicTacToe.attribute_won[0]]
        attr_dict[TicTacToe.attribute_agent[0]] = self.__agent
        attr_dict[TicTacToe.attribute_board[0]] = np.copy(self.__board)
        return attr_dict

    #
    # Is there a winning move on the board.
    #
    def __episode_won(self,
                      board=None) -> bool:
        if board is None:
            board = self.__board
        rows = np.abs(np.sum(board, axis=1))
        cols = np.abs(np.sum(board, axis=0))
        diag_lr = np.abs(np.sum(board.diagonal()))
        diag_rl = np.abs(np.sum(np.rot90(board).diagonal()))

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
    # Are there any remaining actions to be taken
    #
    def __actions_left_to_take(self,
                               board=None):
        if board is None:
            board = self.__board
        return board[np.isnan(board)].size > 0

    #
    # Are there any remaining actions to be taken >
    #
    def __actions_ids_left_to_take(self,
                                   board=None):
        if board is None:
            board = self.__board
        alt = np.reshape(board, board.size)
        alt = np.fromiter(self.actions().keys(), int)[np.isnan(alt)]
        return alt

    #
    # The episode is over if one agent has made a line of three on
    # any horizontal, vertical or diagonal or if there are no actions
    # left to take and neither agent has won.
    #
    def episode_complete(self,
                         state: State = None):

        board = None
        if state is not None:
            board = state.state()

        if self.__episode_won(board) or not self.__actions_left_to_take(board):
            return True
        return False

    #
    # Convert an environment (board) from a string form to the
    # internal board curr_coords.
    #
    def __string_to_internal_state(self, moves_as_str):
        mvs = moves_as_str.split('~')
        if moves_as_str is not None:
            for mv in mvs:
                if len(mv) > 0:
                    pl, ps = mv.split(":")
                    self.__take_action(int(ps), self.__agents[int(pl)])
        return

    #
    # Convert internal (board) curr_coords to string
    #
    def __internal_state_to_string(self, board) -> str:
        mvs = ""
        bd = np.reshape(self.__board, self.__board.size)
        cell_num = 0
        for actor in bd:
            if not np.isnan(actor):
                mvs += str(int(actor)) + ":" + str(int(cell_num)) + "~"
            cell_num += 1
        if len(mvs) > 0:
            mvs = mvs[:-1]
        return mvs

    #
    # The current curr_coords of the environment as string
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
    # Expose current environment curr_coords as string
    #
    def export_state(self):
        return self.__internal_state_to_string(self.__board)

    #
    # Set environment curr_coords from string
    #
    def import_state(self, state_as_string):
        self.__string_to_internal_state(state_as_string)

    #
    # Return the State of the environment
    #
    def state(self) -> State:
        return TicTacToeState(self.__board,
                              self.__x_agent,
                              self.__o_agent)

    # Policy was not linked to an environment before it was used..
    #
    class IllegalActorAction(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    #
    # Randomise Player Turns, if not Random then player X always goes first
    #
    @property
    def random_player_turns(self) -> bool:
        return self.__random_turns

    @random_player_turns.setter
    def random_player_turns(self,
                            value: bool) -> None:
        if type(value) != bool:
            raise TypeError("Player turns is boolean cannot not be type [" + type(value).__name__ + "]")
        self.__random_turns = value
        return
