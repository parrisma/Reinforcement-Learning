import numpy as np
from reflrn.Interface.Telemetry import Telemetry
from reflrn.SimpleStateTelemetry import SimpleStateTelemetry


#
# Track and report on the telemetry for the ActorCriticPolicy
#
class ActorCriticPolicyTelemetry(Telemetry):

    def __init__(self):
        self.__telemetry = dict()

    #
    # Update State Telemetry
    #
    def update_state_telemetry(self,
                               state_as_array: np.ndarray) -> None:
        state_as_str = self.__a2s(state_as_array)
        if state_as_str not in self.__telemetry:
            self.__telemetry[state_as_str] = SimpleStateTelemetry(state_as_str)
        self.__telemetry[state_as_str].frequency = self.__telemetry[state_as_str].frequency + 1
        return

    #
    # Return the telemetry for the given state or None if there no telemetry held for the
    # given state
    #
    def state_observation_telemetry(self,
                                    state_as_arr: np.ndarray) -> 'Telemetry.StateTelemetry':
        state_as_str = self.__a2s(state_as_arr)
        if state_as_str in self.__telemetry:
            return self.__telemetry[state_as_str]
        return 0

    #
    # Array To Str
    #
    @classmethod
    def __a2s(cls, arr: np.ndarray) -> str:
        return np.array2string(np.reshape(arr, np.size(arr)), separator='')
