import abc


#
# This abstract base class that is the contract for communicating Model parameters
#


class ModelParams(metaclass=abc.ABCMeta):
    learning_rate_max = 'learning_rate_max'
    learning_rate_min = 'learning_rate_min'
    learning_rate_0 = 'learning_rate_0'
    learning_rate_decay = 'learning_rate_decay'
    batch_size = 'batch_size'
    epsilon = 'epsilon'
    epsilon_decay = 'epsilon_decay'
    gamma = 'gamma'
    gamma_decay = 'gamma_decay'

    #
    # Getter Methods For model parameters
    #
    @abc.abstractmethod
    def get_parameter(self,
                      params: [str]):
        pass

    #
    # Getter Methods For model parameters
    #
    @abc.abstractmethod
    def override_parameters(self,
                            params: [[str, object]]):
        pass

    # Requested Parameter no available
    #
    class RequestedParameterNotAvailable(Exception):
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)
