#
# Allow hard wired games to be played for testing.
#


class FixedGames:
    game_id = int(0)
    play_id = int(0)
    games = [(0, 8, 2, 6, 4, 3, 7, 5, 1)]

    #
    # Iterate over the fixed games returning the next action.
    #
    @classmethod
    def next_action(cls) -> int:
        if cls.play_id >= len(cls.games[cls.game_id]):
            cls.play_id = 0
            cls.game_id += 1
            if cls.game_id >= len(cls.games)-1:
                cls.game_id = 0

        a = cls.games[cls.game_id][cls.play_id]
        cls.play_id += 1
        return a