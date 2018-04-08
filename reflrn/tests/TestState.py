from State import State


class TestState(State):

    def __init__(self,state_as_str: str):
        self.__state_as_str = state_as_str

    def state(self) -> object:
        return None

    def state_as_string(self) -> str:
        return self.__state_as_str