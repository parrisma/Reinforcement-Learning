from reflrn.Interface.Telemetry import Telemetry


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
                               state_as_str: str) -> None:
        if state_as_str not in self.__telemetry:
            self.__telemetry[state_as_str] = Telemetry.StateTelemetry(state_as_str)
        Telemetry.StateTelemetry(self.__telemetry[state_as_str]).frequency += 1
        return

    #
    # Return the telemetry for the given state or None if there no telemetry held for the
    # given state
    #
    def state_observation_telemetry(self,
                                    state_as_str: str) -> 'Telemetry.StateTelemetry':
        if state_as_str in self.__telemetry:
            return self.__telemetry[state_as_str]
        return None
