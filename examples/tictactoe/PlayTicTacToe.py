import random
from random import randint

import numpy as np

from examples.tictactoe.TicTacToe import TicTacToe


class PlayTicTacToe:
    __learning_rate_0 = 0.05
    __learning_rate_decay = 0.001
    __discount_factor = .8
    __q_values = {}  # learning spans game sessions.

    #
    # Constructor has no arguments as it just sets the game
    # to an initial up-played set-up
    #
    def __init__(self, persist=None):
        self.__game = TicTacToe()
        self.__persist = persist

    #
    # Return the current game
    #
    def game(self):
        return self.__game

    #
    # Set learned curr_coords to given QValues.
    #
    def transfer_learning(self, qv):
        self.__q_values = qv
        print("Learned Games:" + str(len(self.__q_values)))

    #
    # The learned Q Values for a given curr_coords if they exist
    #
    def q_vals_for_state(self, state):
        if state in self.__q_values:
            return self.__q_values[state]
        else:
            return None

    #
    # Expose the current class instance learning in terms of Q Values.
    #
    def q_vals(self):
        return self.__q_values

    #
    # Load Q Val curr_coords
    #
    def load_q_vals(self, filename):
        if self.__persist is not None:
            self.__q_values = self.__persist.load(filename)
        return

    #
    # Save Q Val curr_coords
    #
    def save_q_vals(self, filename):
        if self.__persist is not None:
            self.__persist.save(self.__q_values, filename)
        return

    #
    # Forget learning
    #
    def forget_learning(self):
        self.__q_values = dict()

    #
    # Add states to Q Value dictionary if not present
    #
    def add_states_if_missing(self, state):
        if self.__q_values is None:
            self.__q_values = dict()

        if state not in self.__q_values:
            self.__q_values[state] = TicTacToe.__empty_board()
            self.__q_values[state] = np.reshape(self.__q_values[state], self.__q_values[state].size)  # flatten

    #
    # Return the learning rate paramaters
    #
    @classmethod
    def learning_rate_params(cls):
        return cls.__learning_rate_0, cls.__learning_rate_decay, cls.__discount_factor

    #
    # Return the learning rate based on number of learnings to date
    #
    @classmethod
    def q_learning_rate(cls, n):
        return cls.__learning_rate_0 / (1 + (n * cls.__learning_rate_decay))

    #
    # Return the State, Action Key from the perspective of given player
    #
    @classmethod
    def state(cls, player, board):
        sa = ""
        sa += str(player)
        for cell in np.reshape(board, TicTacToe.num_actions()).tolist():
            sa += str(TicTacToe.player_to_int(cell))
        return sa

    #
    # The optimal action is to take the largest positive gain or the smallest
    # loss. => Maximise Gain & Minimise Loss
    #
    @classmethod
    def optimal_outcome(cls, q):
        q = q[np.isnan(q) == False]
        if len(q) > 0:
            stand_to_win = q[q >= 0]
            stand_to_lose = q[q < 0]
            if stand_to_win.size > 0:
                return np.max(stand_to_win)
            else:
                if stand_to_lose.size > 0:
                    return np.max(stand_to_lose)
                else:
                    return np.nan
        else:
            return np.nan

    #
    # Return zero float if given number is "nan"
    #
    @classmethod
    def zero_if_nan(cls, n):
        if np.isnan(n):
            return 0
        else:
            return n

    #
    # Run simulation to estimate Q values for curr_coords, action pairs. Random exploration policy
    # which should be tractable with approx 6K valid board states. This function takes "canned"
    # moves which were full game sequences created else where.
    #
    def train_q_values(self, num_episodes, canned_moves):

        # Simulation defaults.
        learning_rate0, learning_rate_decay, discount_rate = PlayTicTacToe.learning_rate_params()

        # Initialization
        sim = 0

        # Iterate over and play
        while sim < num_episodes:
            reward = None
            self.__game.reset()
            plyr = None
            prev_plyr = None
            s = None
            mv = None
            prev_mv = None
            prev_s = None
            mv = None

            game_step = 0
            while not self.__game.episode_complete():

                prev_mv = mv
                plyr, mv = (canned_moves[sim])[game_step]
                prev_plyr = TicTacToe.other_player(plyr)
                prev_s = s

                s = PlayTicTacToe.state(plyr, self.__game.board())
                reward = self.__game.__play_action(mv, plyr)
                learning_rate = PlayTicTacToe.q_learning_rate(len(self.__q_values))

                self.add_states_if_missing(s)

                # Update Q Values for both players based on last play reward.
                (self.__q_values[s])[mv - 1] = (learning_rate * (self.zero_if_nan(self.__q_values[s][mv - 1]))) + (
                        (1 - learning_rate) * reward)
                # Update any discounted rewards to previous game step.
                if prev_s is not None:
                    (self.__q_values[prev_s])[prev_mv - 1] -= (discount_rate * self.optimal_outcome(self.__q_values[s]))
                game_step += 1
            sim += 1
            game_step = 0

        return self.__q_values

    #
    # Run simulation to estimate Q values for curr_coords, action pairs. Random exploration policy
    # which should be tractable with approx 6K valid board states.
    #
    def train_q_values_r(self, num_simulations):

        learning_rate0, learning_rate_decay, discount_rate = PlayTicTacToe.learning_rate_params()

        reward = 0
        sim = 0
        game_step = 0

        while sim < num_simulations:
            self.__game.reset()
            plyr = None
            s = None
            mv = None
            prev_mv = None
            prev_s = None

            plyr = (TicTacToe.player_x, TicTacToe.player_o)[randint(0, 1)]  # Random player to start

            mv = None
            while not self.__game.episode_complete():

                prev_mv = mv
                st = PlayTicTacToe.state(plyr, self.__game.board())
                if random.random() > 0.8:
                    mv = self.informed_action(st, False)  # Informed Player
                else:
                    mv = self.informed_action(st, True)  # Random Player

                prev_s = s
                s = PlayTicTacToe.state(plyr, self.__game.board())
                reward = self.__game.__play_action(mv, plyr)
                learning_rate = PlayTicTacToe.q_learning_rate(len(self.__q_values))

                self.add_states_if_missing(s)

                # Update Q Values for both players based on last play reward.
                (self.__q_values[s])[mv - 1] = (learning_rate * (self.zero_if_nan(self.__q_values[s][mv - 1]))) + (
                        (1 - learning_rate) * reward)
                if prev_s is not None:
                    (self.__q_values[prev_s])[prev_mv - 1] -= (discount_rate * self.optimal_outcome(self.__q_values[s]))

                plyr = TicTacToe.other_player(plyr)
                game_step += 1
            sim += 1
            game_step = 0

        return self.__q_values

    #
    # Given current curr_coords and learned Q Values (if any) suggest
    # the play_action that is expected to yield the highest reward.
    #
    def informed_action(self, st, rnd=False, model=None):
        # What moves are possible at this stage
        valid_moves = self.__game.what_are_valid_moves()

        # Are there any moves ?
        if np.sum(valid_moves * np.full(TicTacToe.num_actions(), 1)) == 0:
            return None

        optimal_action = None
        if not rnd:
            # Predict Q Vals if there is a model, else try to find
            # a hit in leanred Q vals.
            if model is not None:
                q_vals = model.predicted_q_vals(st)
            else:
                q_vals = self.q_vals_for_state(st)

            if q_vals is not None:
                q_vals *= valid_moves
                optimal_action = PlayTicTacToe.optimal_outcome(q_vals)
                q_vals = (q_vals == optimal_action) * TicTacToe.actions()
                q_vals = q_vals[np.where(q_vals != 0)]
                if q_vals.size > 0:
                    optimal_action = q_vals[randint(0, q_vals.size - 1)]
                else:
                    optimal_action = None

        # If we found a good action then return that
        # else pick a random action
        #
        if optimal_action is None:
            actions = valid_moves * np.arange(1, TicTacToe.num_actions() + 1, 1)
            actions = actions[np.where(actions > 0)]
            optimal_action = actions[randint(0, actions.size - 1)]

        return int(optimal_action)
        #

    # Play an automated game between a random player and an
    # informed player.
    # Return the play_action sequence for the entire game as s string.
    #
    def play(self):
        self.__game.reset()
        plyr = (TicTacToe.player_x, TicTacToe.player_o)[randint(0, 1)]  # Chose random player to start
        mv = None
        game_moves_as_str = ""
        while not self.__game.episode_complete():
            st = PlayTicTacToe.state(plyr, self.__game.board())
            if plyr == TicTacToe.player_x:
                mv = self.informed_action(st, False)  # Informed Player
            else:
                mv = self.informed_action(st, True)  # Random Player
            self.__game.__play_action(mv, plyr)
            game_moves_as_str += str(plyr) + ":" + str(mv) + "~"
            plyr = TicTacToe.other_player(plyr)
        return game_moves_as_str

    #
    # Add the game profile to the given game dictionary and
    # up the count for the number of times that games was played
    #
    @classmethod
    def record_game_stats(cls, game_stats_dict, profile):
        if profile in game_stats_dict:
            game_stats_dict[profile] += 1
        else:
            game_stats_dict[profile] = 1
        return

    #
    # Play a given number of informed games against a random player and
    # track the relative outcome. Wins/Losses/Draws
    # ToDo: check the win/lss/draw logic as this seems to report zero draws.
    #
    def play_many(self, num):
        informed_wins = 0
        random_wins = 0
        draws = 0
        informed_game = {}
        random_game = {}
        drawn_game = {}
        distinct_games = {}

        for x in range(0, num):
            profile = self.play()
            if profile not in distinct_games:
                distinct_games[profile] = ""
            if self.__game.episode_complete(self.__game.board(), TicTacToe.player_x):
                informed_wins += 1
                PlayTicTacToe.record_game_stats(informed_game, profile)
            else:
                if self.__game.episode_complete(self.__game.board(), TicTacToe.player_o):
                    random_wins += 1
                    PlayTicTacToe.record_game_stats(random_game, profile)
                else:
                    PlayTicTacToe.record_game_stats(drawn_game, profile)
                    draws += 1
            if (x % 100) == 0:
                print(str(x))
        print("Informed :" + str(informed_wins) + " : " + str(round((informed_wins / num) * 100, 0)))
        print("Random :" + str(random_wins) + " : " + str(round((random_wins / num) * 100, 0)))
        print("Draw :" + str(draws) + " : " + str(round((draws / num) * 100, 0)))
        print("Diff Games :" + str(len(distinct_games)))
        return informed_game, random_game, drawn_game

    #
    # Convert a game profile string returned from play method
    # into an array that can be passed as a canned-play_action to
    # training. (Q learn)
    #
    @classmethod
    def string_of_moves_to_array(cls, moves_as_str):
        mvd = {}
        mvc = 0
        mvs = moves_as_str.split('~')
        for mv in mvs:
            if len(mv) > 0:
                pl, ps = mv.split(":")
                mvd[mvc] = (int(pl), int(ps))
            mvc += 1
        return mvd

    #
    # Convert a game profile string returned from play method
    # into an array that can be passed as a canned-play_action to
    # training. (Q learn)
    #
    @classmethod
    def string_of_moves_to_a_board(cls, moves_as_str):
        mvc = 0
        mvs = moves_as_str.split('~')
        bd = np.reshape(TicTacToe.__empty_board(), TicTacToe.num_actions())
        for mv in mvs:
            if len(mv) > 0:
                pl, ps = mv.split(":")
                bd[int(ps) - 1] = int(pl)
            mvc += 1
        return np.reshape(bd, (3, 3))

    #
    # Convert a dictionary of game profiles returned from play_many
    # to a dictionary of canned moves that can be passed to training (Q Learn)
    #
    @classmethod
    def moves_to_dict(cls, move_dict):
        md = {}
        i = 0
        for mvss, cnt in move_dict.items():
            md[i] = PlayTicTacToe.string_of_moves_to_array(mvss)
            i += 1
        return md

    #
    # All possible endings. Generate moves str's for all the possible endings of the
    # game from the perspective of the prev player.
    #
    # The given moves must be the moves of a valid game that played to either win/draw
    # including the last play_action that won/drew the game.
    #
    @classmethod
    def all_possible_endings(cls, moves_as_str, exclude_current_ending=True):
        ape = {}
        mvs = PlayTicTacToe.string_of_moves_to_array(moves_as_str)

        terminal_move = mvs[len(mvs) - 1]  # The play_action that won, drew
        last_move = mvs[len(mvs) - 2]  # the play_action we will replace with all other options

        t_plyr = terminal_move[0]
        t_actn = terminal_move[1]

        l_plyr = last_move[0]
        l_actn = last_move[1]

        base_game = "~".join(moves_as_str.split("~")[:-3])  # less Trailing ~ + terminal & last play_action
        bd = PlayTicTacToe.string_of_moves_to_a_board(base_game)
        vmvs = TicTacToe.valid_moves(bd)
        a = 1
        for vm in vmvs:
            poss_end = base_game
            if vm:
                if a != t_actn:  # don't include the terminal action as we will add that back on.
                    if not (exclude_current_ending and a == l_actn):
                        poss_end += "~" + str(l_plyr) + ":" + str(a)
                        poss_end += "~" + str(t_plyr) + ":" + str(t_actn) + "~"
                        ape[poss_end] = 0
            a += 1
        return ape

    #
    # return single q val as formatted float or spaces for nan
    #
    @classmethod
    def __single_q_value_to_str(cls, sqv):
        if np.sum(np.isnan(sqv) * 1) > 0:
            return " " * 26
        s = '{:+.16f}'.format(sqv)
        s = " " * (26 - len(s)) + s
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
            s += rbd + "    " + rqv + "\n"
        s += "\n"
        return s

    #
    # Make a play based on q values (called via interactive game)
    #
    def machine_move(self, *args):
        st = PlayTicTacToe.state(TicTacToe.player_x, self.game().board())
        qv = self.q_vals_for_state(st)
        print(self.board_as_string(self.game().board(), qv))
        if args is not None:
            model = args[0][0]  # Model
        mv = self.informed_action(st, False, model)
        self.game().__play_action(mv, TicTacToe.player_x)
        return str(TicTacToe.player_x) + ":" + str(mv) + "~"

    #
    # Make a play based on human input  (called via interactive game)
    #
    def human_move(self, *args):
        st = PlayTicTacToe.state(TicTacToe.player_o, self.game().board())
        qv = self.q_vals_for_state(st)
        print(self.board_as_string(self.game().board(), qv))
        mv = input("Make your play_action: ")
        self.game().__play_action(int(mv), TicTacToe.player_o)
        return str(TicTacToe.player_o) + ":" + str(mv) + "~"

    #
    # Play an interactive game with the informed player via
    # stdin.
    #
    def interactive_game(self, human_first, *args):
        self.__game.reset()
        mvstr = ""

        player_move = dict()
        player_move[1] = PlayTicTacToe.machine_move
        player_move[2] = PlayTicTacToe.human_move
        if human_first:
            player_move[1], player_move[2] = player_move[2], player_move[1]

        while not self.__game.episode_complete():
            mvstr += player_move[1](self, args)
            if self.__game.episode_complete():
                break
            mvstr += player_move[2](self, args)

        print(TicTacToe.board_as_string(self.game().board()))
        print("Game Over")

        # Learn from game just played by playing all the possible endings of the
        # winning / losing game.
        ape = PlayTicTacToe.all_possible_endings(mvstr)
        if len(ape) > 0:
            for i in range(0, 5):
                self.train_q_values(len(ape), PlayTicTacToe.moves_to_dict(ape))
