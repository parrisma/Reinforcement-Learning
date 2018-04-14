from .IllegalGridMoveException import IllegalGridMoveException


class GridBlockedActionException(IllegalGridMoveException):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
