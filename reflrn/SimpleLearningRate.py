from reflrn.Interface.LearningRate import LearningRate


#
# Implements a learning rate of lr = lr_zero / (1 + (step * lr_decay_factor))
#


class SimpleLearningRate(LearningRate):

    def __init__(self,
                 lr0: float,
                 lrd: float,
                 lr_min: float = float(0)):
        self.__lr0 = lr0
        self.__lrd = lrd
        self.__lr_min = lr_min
        return

    #
    # What is the learning rate for the given step; with a floor of the min learning rate
    #
    def learning_rate(self,
                      step: int) -> float:
        lr = self.__lr0 / (1 + (step * self.__lrd))
        return max(self.__lr_min, lr)

    #
    # What learning rate decay would give a learning_rate_target (value) after target_step steps ?
    #
    @classmethod
    def lr_decay_target(cls,
                        learning_rate_zero: float,
                        target_step: int,
                        target_learning_rate: float) -> float:
        if target_step <= 0:
            raise ValueError("Taregt Step must be >= 1")
        return (learning_rate_zero - target_learning_rate) / (target_step * target_learning_rate)
