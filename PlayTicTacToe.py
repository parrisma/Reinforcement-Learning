import numpy as np
import pickle
from random import randint
from TicTacToe import TicTacToe

class PlayTicTacToe:

    __learning_rate_0 = 0.05
    __learning_rate_decay = 0.001
    __discount_factor = .8
    __Q = {}  # learning spans game sessions.

    #
    # Constructor has no arguments as it just sets the game
    # to an intial up-played set-up
    #
    def __init__(self):
        self.__game = TicTacToe()

    #
    # Return the current game
    #
    def game(self):
        return self.__game

    #
    # Set leared state to given QValues.
    #
    def transfer_learning(self, QV):
        self.__Q = QV
        print("Learned Games:" + str(len(self.__Q)))

    #
    # The learned Q Values for a given state if they exist
    #
    def Q_Vals_for_state(self, state):
        if (state in self.__Q):
            return self.__Q[state]
        else:
            return None

    #
    # Expose the current class instance learning in terms of Q Values.
    #
    def Q_Vals(self):
        return self.__Q

    #
    # Load Q Val state
    #
    @classmethod
    def load_q_vals(cls, filename):
        cls.__Q = pickle.load(open(filename, "rb"))
        return len(cls.__Q)

    #
    # Save Q Val state
    #
    @classmethod
    def save_q_vals(cls, filename):
        try:
            pickle.dump(cls.__Q, open(filename, "wb"))
            return True
        except Exception as exc:
            print("Failed to save Q Values : " + str(exc))
            return False

    #
    # Forget learning
    #
    def forget_learning(self):
        self.__Q = {}

    #
    # Add states to Q Value dictionary if not present
    #
    def add_states_if_missing(self, s1):
        if s1 not in self.__Q:
            self.__Q[s1] = np.full(TicTacToe.num_actions(), np.nan)

    #
    # Return the learning rate paramaters
    #
    @classmethod
    def learning_rate_params(cls):
        return cls.__learning_rate_0, cls.__learning_rate_decay, cls.__discount_factor

    #
    # Return the learnign rate based on number of learnings to date
    #
    @classmethod
    def q_learning_rate(cls, n):
        return cls.__learning_rate_0 / (1 + (n * cls.__learning_rate_decay))

    #
    # Keep Score of players as Q Val Trains.
    #
    @classmethod
    def __init_score(cls):
        score = dict()
        score[TicTacToe.player_X] = {}
        score[TicTacToe.player_O] = {}
        for rn, rv in TicTacToe.rewards().items():
            score[TicTacToe.player_X][rv] = 0
            score[TicTacToe.player_O][rv] = 0
        return score

    @classmethod
    def __keep_score(cls, score, plyr, reward):
        (score[plyr])[reward[0]] += 1
        (score[TicTacToe.other_player(plyr)])[reward[1]] += 1
        return

    #
    # Return the State, Action Key from the perspective of given player
    #
    @classmethod
    def state(cls, player, board):
        sa = ""
        sa += str(player)
        for cell in np.reshape(board, 9).tolist(): sa += str(TicTacToe.player_to_int(cell))
        return sa

    #
    # Given q values for move to a given state select the
    # as to maximise players gain or if not gain to be had
    # minimise opponents gain
    #
    @classmethod
    def best_outcome(cls, q):
        q = q[np.isnan(q) == False]
        if len(q) > 0:
            stand_to_win = np.max((q >= 0) * q)
            stand_to_lose = -np.min((q < 0) * q)
            return max(stand_to_win, stand_to_lose)
        else:
            return np.nan

    #
    # The best move is the highest gain
    #
    @classmethod
    def best_move(cls, q):
        return cls.best_outcome(q)

    #
    # Return zero float if given number is "nan"
    #
    @classmethod
    def zero_if_nan(cls, n):
        if np.isnan(n):
            return 0
        else:
            return n

    # Run simulation to estimate Q values for state, action pairs. Random exploration policy
    # which should be tractable with approx 6K valid board states.
    #
    def train_Q_values(self, num_episodes, canned_moves):

        # Simulation defaults.
        learning_rate0, learning_rate_decay, discount_rate = PlayTicTacToe.learning_rate_params()

        # Initalization
        reward = 0
        sim = 0
        game_step = 0
        score = PlayTicTacToe.__init_score()

        # Iterate over and play
        while (sim < num_episodes):
            self.__game.reset()
            plyr = None
            prev_plyr = None
            s = None
            mv = None
            prev_mv = None
            prev_s = None
            mv = None

            game_step = 0
            while (not self.__game.game_over()):

                prev_mv = mv
                print(str(sim) + " : " + str(game_step))
                plyr, mv = (canned_moves[sim])[game_step]
                prev_plyr = TicTacToe.other_player(plyr)
                prev_s = s

                s = PlayTicTacToe.state(plyr, self.__game.board())
                reward = self.__game.move(mv, plyr)
                learning_rate = PlayTicTacToe.q_learning_rate(len(self.__Q))

                self.add_states_if_missing(s)

                # Update Q Values for both players based on last play reward.
                (self.__Q[s])[mv - 1] = (learning_rate * (self.zero_if_nan(self.__Q[s][mv - 1]))) + ((1 - learning_rate) * reward[0])
                if (not prev_s is None):
                    (self.__Q[prev_s])[prev_mv - 1] -= (discount_rate * self.best_outcome(self.__Q[s]))
                game_step += 1
            sim += 1
            game_step = 0

            PlayTicTacToe.__keep_score(score, plyr, reward)

            if ((sim % 1000) == 0) or (sim == num_episodes):
                smX = "Player X : " + str(sim) + " : "
                smO = "Player O : " + str(sim) + " : "
                for rn, rv in TicTacToe.rewards().items():
                    smX += rn + " : " + str(round(((score[TicTacToe.player_X])[rv] / sim) * 100, 0)) + "% "
                    smO += rn + " : " + str(round(((score[TicTacToe.player_O])[rv] / sim) * 100, 0)) + "% "
                print(smX)
                print(smO)
        return self.__Q

    #
    # Run simulation to estimate Q values for state, action pairs. Random exploration policy
    # which should be tractable with approx 6K valid board states.
    #
    def train_Q_values_R(self, num_simulations):

        learning_rate0, learning_rate_decay, discount_rate = PlayTicTacToe.learning_rate_params()

        reward = 0
        sim = 0
        game_step = 0
        score = PlayTicTacToe.__init_score()

        while (sim < num_simulations):
            self.__game.reset()
            plyr = None
            s = None
            mv = None
            prev_mv = None
            prev_s = None

            plyr = (TicTacToe.player_X, TicTacToe.player_O)[randint(0, 1)]  # Random player to start

            mv = None
            while (not self.__game.game_over()):

                prev_mv = mv
                mv = self.random_move()

                prev_s = s
                s = PlayTicTacToe.state(plyr, self.__game.board())
                reward = self.__game.move(mv, plyr)
                learning_rate = PlayTicTacToe.q_learning_rate(len(self.__Q))

                self.add_states_if_missing(s)

                # Update Q Values for both players based on last play reward.
                (self.__Q[s])[mv - 1] = (learning_rate * (self.zero_if_nan(self.__Q[s][mv - 1]))) + ((1 - learning_rate) * reward[0])
                if (not prev_s is None):
                    (self.__Q[prev_s])[prev_mv - 1] -= (discount_rate * self.best_outcome(self.__Q[s]))

                plyr = TicTacToe.other_player(plyr)
                game_step += 1
            sim += 1
            game_step = 0

            PlayTicTacToe.__keep_score(score, plyr, reward)

            if ((sim % 1000) == 0) or (sim == num_simulations):
                smX = "Player X : " + str(sim) + " : "
                smO = "Player O : " + str(sim) + " : "
                for rn, rv in TicTacToe.rewards().items():
                    smX += rn + " : " + str(round(((score[TicTacToe.player_X])[rv] / sim) * 100, 0)) + "% "
                    smO += rn + " : " + str(round(((score[TicTacToe.player_O])[rv] / sim) * 100, 0)) + "% "
                print(smX)
                print(smO)
        return self.__Q

    #
    # Return a random action (move) that is still left
    # to make
    #
    def random_move(self):
        valid_moves = None
        random_action = None

        valid_moves = np.isnan(self.__game.board().reshape(9))*self.__game.actions()
        valid_moves = valid_moves[np.where(valid_moves>0)]

        num_poss_moves = len(valid_moves)
        if num_poss_moves > 0:
            random_action = valid_moves[randint(0, num_poss_moves - 1)]
            return random_action
        else:
            return None

    #
    # Given current state and lerned Q Values (if any) suggest
    # the move that is expected to yield the highest reward.
    #
    def informed_move(self, st, rnd):
        # What moves are possible at this stage
        valid_moves = self.__game.what_are_valid_moves()

        # Are there any moves ?
        if (np.sum(valid_moves * np.full(9, 1)) == 0):
            return None

        best_action = None
        if (not rnd):
            # Is there info learned for this state ?
            informed_actions = self.Q_Vals_for_state(st)
            if not informed_actions is None:
                informed_actions *= valid_moves
                best_action = PlayTicTacToe.best_move(informed_actions)
                informed_actions = (informed_actions==best_action)*TicTacToe.actions()
                informed_actions = informed_actions[np.where(informed_actions!=0)]
                if informed_actions.size > 0:
                    best_action = informed_actions[randint(0, informed_actions.size - 1)]
                else:
                    best_action = None

        # If we found a good action then return that
        # else pick a random action
        if best_action == None:
            actions = valid_moves * np.arange(1, TicTacToe.num_actions() + 1, 1)
            actions = actions[np.where(actions > 0)]
            best_action = actions[randint(0, actions.size - 1)]

        return int(best_action)
        #

    # Play an automated game between a random player and an
    # informed player.
    # Return the move sequence for the entire game as s string.
    #
    def play(self):
        self.__game.reset()
        plyr = (TicTacToe.player_X, TicTacToe.player_O)[randint(0, 1)]  # Chose random player to start
        mv = None
        profile = ""
        while (not self.__game.game_over()):
            st = PlayTicTacToe.state(plyr, self.__game.board())
            QV = self.Q_Vals_for_state(st)
            mx = np.max(self.Q_Vals_for_state(st))
            if (plyr == TicTacToe.player_X):
                mv = self.informed_move(st, False)  # Informed Player
            else:
                mv = self.informed_move(st, True)  # Random Player
            self.__game.move(mv, plyr)
            profile += str(plyr) + ":" + str(mv) + "~"
            plyr = TicTacToe.other_player(plyr)
        return profile

    #
    # Add the game profile to the given game dictionary and
    # up the count for the number of times that games was played
    #
    @classmethod
    def record_game_stats(cls, D, profile):
        if profile in D:
            D[profile] += 1
        else:
            D[profile] = 1
        return

    def play_many(self, num):
        informed_wins = 0
        random_wins = 0
        draws = 0
        I = {}
        R = {}
        D = {}
        G = {}
        profile = ""
        for x in range(0, num):
            profile = self.play()
            if profile not in G: G[profile] = ""
            if self.__game.game_won(self.__game.board(), TicTacToe.player_X):
                informed_wins += 1
                PlayTicTacToe.record_game_stats(I, profile)
            else:
                if self.__game.game_won(self.__game.board(), TicTacToe.player_O):
                    random_wins += 1
                    PlayTicTacToe.record_game_stats(R, profile)
                else:
                    PlayTicTacToe.record_game_stats(D, profile)
                    draws += 1
            if (x % 100) == 0: print (str(x))
        print("Informed :" + str(informed_wins) + " : " + str(round((informed_wins / num) * 100, 0)))
        print("Random :" + str(random_wins) + " : " + str(round((random_wins / num) * 100, 0)))
        print("Draw :" + str(draws) + " : " + str(round((draws / num) * 100, 0)))
        print("Diff Games :" + str(len(G)))
        return (I, R, D)

    #
    # move_str is of form "1:8~-1:1~1:6~-1:3~1:9~-1:2~"
    # plyr:action~.. repreat players must be alternate X,O (1,-1..)
    # there is always a trailing ~

    #
    # Convert a game profile string returned from play method
    # into an array that can be passed as a canned-move to
    # training. (Q learn)
    #
    @classmethod
    def move_str_to_array(cls, moves_as_str):
        mvd = {}
        mvc = 0
        mvs = moves_as_str.split('~')
        for mv in mvs:
            if (len(mv) > 0):
                pl, ps = mv.split(":")
                mvd[mvc] = (int(pl), int(ps))
            mvc += 1
        return mvd

    #
    # Convert a game profile string returned from play method
    # into an array that can be passed as a canned-move to
    # training. (Q learn)
    #
    @classmethod
    def move_str_to_board(cls, moves_as_str):
        mvd = {}
        mvc = 0
        mvs = moves_as_str.split('~')
        bd = np.zeros((3 * 3), np.int8)
        for mv in mvs:
            if (len(mv) > 0):
                pl, ps = mv.split(":")
                bd[int(ps) - 1] = int(pl)
            mvc += 1
        return np.reshape(bd, (3, 3))

    #
    # Convert a dictionary of game profiles returned from play_many
    # to a dictionary of canned moves that can be passed to training (Q Learn)
    #
    @classmethod
    def moves_to_dict(cls, D):
        MD = {}
        i = 0
        for mvss, cnt in D.items():
            MD[i] = PlayTicTacToe.move_str_to_array(mvss)
            i += 1
        return MD

    #
    # All possible endings. Generate moves str's for all the possible endings of the
    # game from the perspective of the prev player.
    #
    # The given moves must be the moves of a valid game that played to either win/draw
    # including the last move that won/drew the game.
    #
    @classmethod
    def all_possible_endings(cls, moves_as_str, exclude_current_ending=True):
        APE = {}
        mvs = PlayTicTacToe.move_str_to_array(moves_as_str)

        terminal_move = mvs[len(mvs) - 1]  # The move that won, drew
        last_move = mvs[len(mvs) - 2]  # the move we will replace with all other options

        t_plyr = terminal_move[0]
        t_actn = terminal_move[1]

        l_plyr = last_move[0]
        l_actn = last_move[1]

        base_game = "~".join(moves_as_str.split("~")[:-3])  # less Trailing ~ + terminal & last move
        bd = PlayTicTacToe.move_str_to_board(base_game)
        vmvs = TicTacToe.valid_moves(bd)
        a = 1
        for vm in vmvs:
            poss_end = base_game
            if (vm):
                if (a != t_actn):  # don't include the terminal action as we will add that back on.
                    if (not (exclude_current_ending and a == l_actn)):
                        poss_end += "~" + str(l_plyr) + ":" + str(a)
                        poss_end += "~" + str(t_plyr) + ":" + str(t_actn) + "~"
                        APE[poss_end] = 0
            a += 1

        return (APE)


    def interactive_game(self):
        self.__game.reset()
        while not self.__game.game_over():
            print(self.game().board())
            mv = input("Make your move: ")
            self.game().move(int(mv), TicTacToe.player_O)
            st = PlayTicTacToe.state(TicTacToe.player_X, self.game().board())
            qv = self.Q_Vals_for_state(st)
            print("Entry State :" + str(st))
            print("Q Vals: " + str(qv))
            self.game().move(self.informed_move(st, False), TicTacToe.player_X)
        print("Game Over")
        #APE = PlayTicTacToe.all_possible_endings(st)
        #if len(APE) > 0:
        #    qv = self.train_Q_values(len(APE), PlayTicTacToe.moves_to_dict(APE))



