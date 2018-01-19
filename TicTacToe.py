import numpy as np
from Environment import Environment
from Agent import Agent


class TicTacToe(Environment):
    # There are 5812 legal board states that can be reached before there is a winner
    # http://brianshourd.com/posts/2012-11-06-tilt-number-of-tic-tac-toe-boards.html

    __bad_move_game_is_over = -1
    __bad_move_action_already_played = -2
    __bad_move_no_consecutive_plays = -3
    __play = float(-10)  # reward for playing an action
    __bad_play = float(-500)  # reward for taking an action in a cell that has already been played or for trying to take consecutive plays
    __draw = float(0)  # reward for playing to end but no one wins
    __win = float(100)  # reward for winning a game
    sumry_draw = "Draw"
    sumry_won = "Won"
    sumry_actor = "actor"
    __rewards = {"Play": 0, "Draw": 200, "Win": 100, "Loss": -200}
    __no_agent = Agent(np.nan, "")  # id of a non existent player i.e. used to record id of player that has not played
    __win_mask = np.full((1, 3), 3, np.int8)
    __actions = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2)}
    empty_cell = np.nan  # value of a free action space on board
    asStr = True

    #
    # Constructor has no arguments as it just sets the game
    # to an initial up-played set-up
    #
    def __init__(self, x: Agent, o: Agent):
        self.__board = TicTacToe.empty_board()
        self.__last_board = None
        self.__episode_over = False
        self.__episode_drawn = False
        self.__agent = TicTacToe.__no_agent
        self.__last_agent = TicTacToe.__no_agent
        self.__x_agent = x
        self.__o_agent = o

    #
    # Run the given number of iterations
    #
    def run(self, iterations: int):
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
    # Return game to initial state, where no one has played
    # and the board contains no moves.
    #
    def reset(self):
        self.__board = TicTacToe.empty_board()
        self.__last_board = None
        self.__episode_over = False
        self.__episode_drawn = False
        self.__agent = TicTacToe.__no_agent
        self.__last_agent = TicTacToe.__no_agent

    #
    # Return a displayable version of the entire game.
    #
    def __str__(self):
        s = ""
        s += "Episode Over: " + str(self.__episode_over) + "\n"
        s += "Current Agent :" + TicTacToe.__agent_to_str(self.__agent) + "\n"
        s += "Prev Agent :" + TicTacToe.__agent_to_str(self.__last_agent) + "\n"
        s += "Current Board : \n" + str(self.__board) + "\n"
        return s

    #
    # Return the number of possible actions as a list of integers.
    #
    @classmethod
    def num_actions(cls):
        return len(TicTacToe.__actions)

    #
    # Return the actions as a list of integers.
    #
    @classmethod
    def actions(cls):
        return list(map(lambda a: int(a), list(TicTacToe.__actions.keys())))

    #
    # Return the board index (i,j) of a given action
    #
    @classmethod
    def __board_index(cls, action):
        return TicTacToe.__actions[action]

    #
    # Assume the play_action has been validated by play_action method
    # Make a copy of board before play_action is made and the last player
    #
    def __take_action(self, action, agent: Agent):
        self.__last_board = np.copy(self.__board)
        self.__last_agent = self.__agent
        self.__agent = agent.id()
        self.__board[TicTacToe.__board_index(action)] = self.__agent
        return

    #
    # If the proposed action is a valid action and the game is not
    # over. Make the given play_action (action) on behalf of the given
    # player and update the game status.
    #
    # return the rewards (Player who took play_action, Observer)
    #
    def play_action(self, action, agent: Agent):
        #
        # ToDo: This needs a re-work either throw exception and/or treat these as. At the
        # ToDo: very least these need to be thrown as exceptions
        #       things the game needs to learn about actions
        if TicTacToe.episode_complete()[self.sumry_won]:
            raise RuntimeError("Episode is complete, no more actions can be taken")
        if agent == self.__agent:
            raise RuntimeError("Same Agent" + str(agent) + " cannot play consecutive action")

        if self.invalid_action(action):
            return TicTacToe.__bad_move_action_already_played

        # Make the play on the board.
        self.__take_action(action, agent)

        if self.episode_complete():
            es = self.episode_summary()
            if es[self.self.sumry_won]:
                return TicTacToe.__win
            if es[self.self.sumry_draw]:
                return TicTacToe.__draw

        return TicTacToe.__play

    #
    # Show return the current board contents
    #
    def board(self):
        return self.__board

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
    # The episode is over if one agent has made a line of three on
    # any horizontal, vertical or diagonal or if there are no actions
    # left to take and neither agent has won.
    #
    # ToDo: Implementation needs a serious over hall.
    #
    def episode_complete(self):
        if self.__episode_won() or not self.__actions_left_to_take():
            return True
        return False


    #
    # Are there any remaining actions to be taken >
    #
    def __actions_left_to_take(self):
        return self.__board[np.isnan(self.__board)].size > 0

    #
    # Return which player goes next given the current player
    #
    @staticmethod
    def other_player(current_player):
        if current_player == TicTacToe.player_o:
            return TicTacToe.player_x
        else:
            return TicTacToe.player_o

    #
    # What moves are valid for the given board
    #
    @classmethod
    def valid_moves(cls, board):
        vm = np.isnan(board.reshape(TicTacToe.num_actions()))
        return vm

    #
    # What moves are valid given for board in it's current game state
    #
    def what_are_valid_moves(self):
        return TicTacToe.valid_moves(self.__board)

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
    # The current state of the environment as numpy array
    #
    def state_as_array(self) -> np.array():
        return np.reshape(self.__board, self.__board.size)

    #
    # The current state of the environment as string
    #
    def state_as_str(self) -> str:
        return self.__internal_state_to_string()

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
