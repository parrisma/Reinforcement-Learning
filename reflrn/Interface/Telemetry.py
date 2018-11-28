import abc


#
# This abstract class holds telemetry for the learning process
#


class Telemetry(metaclass=abc.ABCMeta):

    #
    # Telemetry on how the given state appeared in the policy training
    #
    @abc.abstractmethod
    def state_observation_telemetry(self,
                                    state_as_str: str) -> 'Telemetry.StateTelemetry':
        pass

    #
    # Telemetry for a specific State
    #
    class StateTelemetry(metaclass=abc.ABCMeta):

        def __init__(self,
                     state_as_str: str):
            self.__state_as_str = state_as_str
            self.__frequency = 0
            return

        def __str__(self) -> str:
            return self.__state.state_as_string() + " observed [" + str(self.__frequency) + "] times during training"

        @property
        @abc.abstractmethod
        def state(self) -> str:
            return self.__state

        @property
        @abc.abstractmethod
        def frequency(self) -> int:
            return self.__frequency

        @frequency.setter
        @abc.abstractmethod
        def frequency(self, value: int):
            self.__frequency = value
