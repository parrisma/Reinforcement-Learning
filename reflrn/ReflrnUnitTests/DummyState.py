import numpy as np

from reflrn.Interface.State import State


class DummyState(State):

    def __init__(self, state_as_str: str):
        self.__state_as_str = state_as_str

    def state(self) -> object:
        return None

    def state_as_string(self) -> str:
        return self.__state_as_str

    def state_as_array(self) -> np.ndarray:
        raise NotImplementedError("state_as_array not implemented for this test DummyState class")
