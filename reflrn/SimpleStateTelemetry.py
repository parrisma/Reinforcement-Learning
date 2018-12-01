from reflrn.Interface.Telemetry import Telemetry


class SimpleStateTelemetry(Telemetry.StateTelemetry):

    def __init__(self,
                 state_as_str: str):
        self.__state_as_str = state_as_str
        self.__frequency = 0
        return

    def __str__(self) -> str:
        return self.__state_as_str + " observed [" + str(self.__frequency) + "] times during training"

    @property
    def state(self) -> str:
        return self.__state_as_str

    @property
    def frequency(self) -> int:
        return self.__frequency

    @frequency.setter
    def frequency(self, value: int):
        self.__frequency = value
