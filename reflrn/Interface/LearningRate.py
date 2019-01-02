import abc


#
# For episodic learning we need to have a learning rate that is a function of the iteration.
#

class LearningRate(metaclass=abc.ABCMeta):

    #
    # Learning rate for the give learning iteration step
    #
    @abc.abstractmethod
    def learning_rate(self,
                      step: int) -> float:
        pass

    #
    # The learning rate decay that will give the target learning rate at the given iteration id
    #
    @abc.abstractmethod
    def lr_decay_target(self,
                        learning_rate_zero: float,
                        target_step: int,
                        target_learning_rate: float) -> float:
        pass
