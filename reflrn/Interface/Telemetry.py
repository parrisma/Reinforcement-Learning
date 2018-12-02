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
    # Export the Telemetry data to the give file.
    #
    @abc.abstractmethod
    def save(self,
             filename: str) -> None:
        raise NotImplementedError

    #
    # Telemetry for a specific State
    #
    class StateTelemetry(metaclass=abc.ABCMeta):

        @property
        @abc.abstractmethod
        def state(self) -> str:
            raise NotImplementedError

        @property
        @abc.abstractmethod
        def frequency(self) -> int:
            raise NotImplementedError

        @frequency.setter
        @abc.abstractmethod
        def frequency(self, value: int):
            raise NotImplementedError
