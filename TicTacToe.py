import numpy as np
from Environment import Environment


class TicTacToe(Environment):

    # There are 5812 legal board states that can be reached before there is a winner
    # http://brianshourd.com/posts/2012-11-06-tilt-number-of-tic-tac-toe-boards.html

    __bad_move_game_is_over = -1
    __bad_move_action_already_played = -2
    __bad_move_no_consecutive_plays = -3
    __play = float(0)  # reward for playing an action
    __draw = float(200)  # reward for playing to end but no one wins
    __win = float(100)  # reward for winning a game
    sumry_draw = "Draw"
    sumry_won = "Won"
    __rewards = {"Play": 0, "Draw": 200, "Win": 100, "Loss": -200}
    __no_player = np.nan  # id of a non existent player i.e. used to record id of player that has not played
    __win_mask = np.full((1, 3), 3, np.int8)
    __actions = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2)}
    player_X = 1  # numerical value of player X on the board
    player_O = -1  # numerical value of player O on the board
    empty_cell = np.nan  # value of a free action space on board
    asStr = True

    #
    # Constructor has no arguments as it just sets the game
    # to an initial up-played set-up
    #
    def __init__(self):
        self.__board = TicTacToe.empty_board()
        self.__last_board = TicTacToe.empty_board()
        self.__game_over = False
        self.__game_drawn = False
        self.__player = TicTacToe.__no_player
        self.__last_player = TicTacToe.__no_player

    #
    # Return a new empty board.
    #
    @classmethod
    def empty_board(cls):
        return np.full((3, 3), np.nan)

    #
    # Return game to initial state, where no one has played
    # and the board contains no moves.
    #
    def reset(self):
        self.__board = TicTacToe.empty_board()
        self.__last_board = TicTacToe.empty_board()
        self.__game_over = False
        self.__game_drawn = False
        self.__player = TicTacToe.__no_player
        self.__last_player = TicTacToe.__no_player

    #
    # Return a displayable version of the entire game.
    #
    def __str__(self):
        s = ""
        s += "Game Over: " + str(self.__game_over) + "\n"
        s += "Player :" + TicTacToe.__player_to_str(self.__player) + "\n"
        s += "Current Board : \n" + str(self.__board) + "\n"
        s += "Prev Player :" + TicTacToe.__player_to_str(self.__last_player) + "\n"
        s += "Prev Current Board : \n" + str(self.__last_board) + "\n"
        return s

    #
    # Render the board as human readable with q values adjacent if supplied
    #
    @classmethod
    def board_as_string(cls, bd, qv=None):
        s = ""
        if qv is not None:
                qv = np.reshape(qv, (3, 3))
        for i in range(0, 3):
            rbd = ""
            rqv = ""
            for j in range(0, 3):
                rbd += "["
                rbd += cls.__player_to_str(bd[i][j])
                rbd += "]"
                if qv is not None:
                    rqv += "["
                    rqv += cls.__single_q_value_to_str(qv[i][j])
                    rqv += "]"
            s += rbd+"    "+rqv+"\n"
        s += "\n"
        return s

    #
    # return player as string "X" or "O"
    #
    @classmethod
    def __player_to_str(cls, player):
        if np.sum(np.isnan(player)*1) > 0:
            return " "
        if player == TicTacToe.player_X:
            return "X"
        if player == TicTacToe.player_O:
            return "O"
        return " "

    #
    # return single q val as formatted float or spaces for nan
    #
    @classmethod
    def __single_q_value_to_str(cls, sqv):
        if np.sum(np.isnan(sqv)*1) > 0:
                return " " * 26
        s = '{:+.16f}'.format(sqv)
        s = " "*(26-len(s))+s
        return s

    #
    # return player as integer
    #
    @classmethod
    def player_to_int(cls, player):
        if np.isnan(player):
            return 0
        return int(player)

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
    def board_index(cls, action):
        return TicTacToe.__actions[action]

    #
    # Return rewards as dictionary where key is name of reward
    # and the value is the reward
    #
    @classmethod
    def rewards(cls):
        return TicTacToe.__rewards

    #
    # Assume the play_action has been validated by play_action method
    # Make a copy of board before play_action is made and the last player
    #
    def __take_action(self, action, player):
        self.__last_board = np.copy(self.__board)
        self.__last_player = self.__player
        self.__player = player
        self.__board[TicTacToe.board_index(action)] = player
        return

    #
    # Has a player already moved using the given action.
    #
    def invalid_action(self, action, board=None):
        if board is None:
            board = self.__board
        return not np.isnan(board[TicTacToe.board_index(action)])

    #
    # If the proposed action is a valid action and the game is not
    # over. Make the given play_action (action) on behalf of the given
    # player and update the game status.
    #
    # return the rewards (Player who took play_action, Observer)
    #
    def play_action(self, action, agent):
        #
        # ToDo: This needs a re-work either throw exception and/or treat these as. At the
        # ToDo: very least these need to be thrown as exceptions
        #       things the game needs to learn about actions
        if TicTacToe.episode_complete()[self.sumry_won]:
                return TicTacToe.__bad_move_game_is_over
        if self.invalid_action(action):
                return TicTacToe.__bad_move_action_already_played
        if agent == self.__player:
                return TicTacToe.__bad_move_no_consecutive_plays

        self.__take_action(action, agent)

        if TicTacToe.episode_complete()[self.sumry_won]:
            self.__game_over = True
            self.__game_drawn = False
            return np.array([TicTacToe.__win, TicTacToe.other_player(agent)])

        if not TicTacToe.__actions_left_to_take(self.__board)[self.sumry_draw]:
            self.__game_over = True
            self.__game_drawn = True
            return np.array([TicTacToe.__draw, TicTacToe.other_player(agent)])

        return np.array([TicTacToe.__play, TicTacToe.other_player(agent)])

    #
    # Show return the current board contents
    #
    def board(self):
        return self.__board


    #
    # Create episode summary; return a dictionary of the episode
    # summary populated with defaults
    #
    @classmethod
    def __episode_summary(cls):
        es = dict()
        es[cls.sumry_draw] = False
        es[cls.sumry_won] = False
        return es

    #
    # The episode is over if one agent has made a line of three on
    # any horizontal, vertical or diagonal or if there are no actions
    # left to take and neither agent has won.
    #
    # ToDo: Implementation needs a serious over hall.
    #
    def episode_complete(self):

        es = self.__episode_summary()

        rows = np.abs(np.sum(self.__board, axis=1))
        cols = np.abs(np.sum(self.__board, axis=0))
        diag_lr = np.abs(np.sum(self.__board.diagonal()))
        diag_rl = np.abs(np.sum(np.rot90(self.__board).diagonal()))

        if np.sum(rows == 3) > 0:
            es[self.sumry_won] = True
            return es
        if np.sum(cols == 3) > 0:
            es[self.sumry_won] = True
            return es
        if not np.isnan(diag_lr):
            if ((np.mod(diag_lr, 3)) == 0) and diag_lr > 0:
                es[self.sumry_won] = True
                return es
        if not np.isnan(diag_rl):
            if ((np.mod(diag_rl, 3)) == 0) and diag_rl > 0:
                es[self.sumry_won] = True
                return es
        if not TicTacToe.__actions_left_to_take(self.__board):
            es[self.sumry_won] = True
            es[self.sumry_draw] = True
            return es

        return es

    #
    # Are there any remaining actions to be taken >
    #
    @classmethod
    def __actions_left_to_take(cls, bd):
        return bd[np.isnan(bd)].size > 0

    #
    # Return which player goes next given the current player
    #
    @staticmethod
    def other_player(current_player):
        if current_player == TicTacToe.player_O:
            return TicTacToe.player_X
        else:
            return TicTacToe.player_O

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

    def load(self, file_name):
        raise NotImplementedError("load() not implemented for TicTacTo")

    def save(self, file_name):
        raise NotImplementedError("save() not implemented for TicTacTo")

    def export_state(self):
        raise NotImplementedError("export_state() not implemented for TicTacTo")

    def import_state(self, state_as_string):
        raise NotImplementedError("import_state() not implemented for TicTacTo")
    