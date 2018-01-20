import numpy as np
from random import randint
from Environment import Environment
from Agent import Agent
from TicTacToeState import TicTacToeState


class TicTacToe(Environment):

    # There are 5812 legal board states that can be reached before there is a winner
    # http://brianshourd.com/posts/2012-11-06-tilt-number-of-tic-tac-toe-boards.html

    __play = float(0)  # reward for playing an action
    __bad_play = float(-500)  # reward for taking an action in a cell that has already been played
    __draw = float(0)  # reward for playing to end but no one wins
    __win = float(100)  # reward for winning a game
    __no_agent = None
    __win_mask = np.full((1, 3), 3, np.int8)
    __actions = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}
    empty_cell = np.nan  # value of a free action space on board
    asStr = True
    sumry_draw = "Draw"
    sumry_won = "Won"
    sumry_actor = "actor"

    #
    # Constructor has no arguments as it just sets the game
    # to an initial up-played set-up
    #
    def __init__(self, x: Agent, o: Agent):
        self.__board = TicTacToe.empty_board()
        self.__last_board = None
        self.__agent = TicTacToe.__no_agent
        self.__last_agent = TicTacToe.__no_agent
        self.__x_agent = x
        self.__o_agent = o
        self.__next_agent = {x.name(): o, o.name(): x}
        self.__x_agent.session_init(self.actions())
        self.__o_agent.session_init(self.actions())

    #
    # Return game to initial state, where no one has played
    # and the board contains no moves.
    #
    def reset(self):
        self.__board = TicTacToe.empty_board()
        self.__last_board = None
        self.__agent = TicTacToe.__no_agent
        self.__last_agent = TicTacToe.__no_agent

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
        while i <= iterations:
            self.reset()
            state = TicTacToeState(self)
            self.__x_agent.episode_init(state)
            self.__o_agent.episode_init(state)
            agent = (self.__x_agent, self.__o_agent)[randint(0, 1)]
            while not self.episode_complete():
                agent = self.__play_action(agent)
                i += 1
            state = TicTacToeState(self)
            self.__x_agent.episode_complete(state)
            self.__o_agent.episode_complete(state)
        return

    #
    # Return a new empty board.
    #
    @classmethod
    def empty_board(cls):
        return np.full((3, 3), np.nan)

    @classmethod
    def no_agent(cls):
        return cls.__no_agent

    #
    # Return the actions as a list of integers.
    #
    @classmethod
    def actions(cls) -> [int]:
        return list(map(lambda a: int(a), list(TicTacToe.__actions.keys())))

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
    # Return True if the given action has already been
    # taken on the board.
    #
    def __invalid_action(self, action: int) -> bool:
        return not np.isnan(self.__board[action])

    #
    # Make the play chosen by the given agent. If it is a valid play
    # confer reward and switch play to other agent. If invalid play
    # i.e. play in a cell where there is already a marker confer
    # penalty and leave play with the same agent.
    #
    def __play_action(self, agent: Agent) -> Agent:

        other_agent = self.__next_agent[agent.name()]
        state = TicTacToeState(self)
        action = self.__actions[agent.chose_action(state)]

        if self.__invalid_action(action):
            agent.reward(state, self.__bad_play)
            return agent

        # Make the play on the board.
        self.__take_action(action, agent)

        if self.episode_complete():
            es = self.episode_summary()
            if es[self.sumry_won]:
                agent.reward(state, self.__win)
                other_agent.reward(state, -1 * self.__win)
                return None
            if es[self.sumry_draw]:
                agent.reward(state, self.__draw)
                other_agent.reward(state, -1 * self.__draw)
                return None

        agent.reward(state, self.__play)
        other_agent.reward(state, -self.__play)
        return other_agent

    #
    # Create episode summary; return a dictionary of the episode
    # summary populated with defaults
    #
    def episode_summary(self):
        es = dict()
        es[self.sumry_draw] = not self.__actions_left_to_take()
        es[self.sumry_won] = self.__episode_won()
        es[self.sumry_actor] = self.__agent
        return es

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
