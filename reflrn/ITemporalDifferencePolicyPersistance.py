import abc
from typing import Tuple

import numpy as np


#
# Interface specification for Temporal Difference Policy Persistance
#


class ITemporalDifferencePolicyPersistance(metaclass=abc.ABCMeta):

    #
    # Dump the given q values dictionary to a simple text dump.
    #
    @classmethod
    def save(cls, qv: dict,
             n: int,
             learning_rate_0: np.float,
             discount_factor: np.float,
             learning_rate_decay: np.float,
             filename: str):
        pass

    #
    # Load the given file into a TD Policy state/action/q value dictionary
    #
    @classmethod
    def load(cls,
             filename: str) -> Tuple[dict, int, np.float, np.float, np.float]:
        pass
