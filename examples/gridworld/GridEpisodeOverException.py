from .IllegalGridMoveException import IllegalGridMoveException


class GridEpisodeOverException(IllegalGridMoveException):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
