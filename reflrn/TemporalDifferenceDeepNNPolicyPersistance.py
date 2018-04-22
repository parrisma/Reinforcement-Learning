import logging
import unittest
from pathlib import Path
from typing import Tuple

import keras
import numpy as np

from reflrn.EnvironmentLogging import EnvironmentLogging
from reflrn.TemporalDifferenceQValPolicyPersistance import TemporalDifferenceQValPolicyPersistance


#
# Save and Load the Keras Model, as well as load the TemporalDifferencePolicy dump file used to train this model.
#


class TemporalDifferenceDeepNNPolicyPersistance:

    def __init__(self, lg: logging):
        self.__lg = lg
        self.__temporal_difference_policy_persistance = TemporalDifferenceQValPolicyPersistance()
        return

    #
    # Dump the entire Keras model to the given filename
    #
    def save(self, model: keras.models.Sequential, filename: str):
        try:
            model.save(filename)
        except Exception as exc:
            err = "Failed to save Keras model to file [" + filename + ": " + str(exc)
            self.__lg.error(err)
            raise RuntimeError(err)
        finally:
            pass
        return True

    #
    # Load the given file into a TD Policy state/action/q value dictionary
    #
    def load(self, filename: str) -> keras.models.Sequential:
        try:
            if Path(filename).is_file():
                model = keras.models.load_model(filename)
            else:
                raise FileExistsError("Keras Model File: [" + filename + "] Not found")
        except Exception as exc:
            err = "Failed to load Keras Deep NN model from file [" + filename + ": " + str(exc)
            self.__lg.error(err)
            raise RuntimeError(err)
        finally:
            pass
        return model

    #
    # Convert a state string to numpy vector
    #
    @classmethod
    def state_as_str_to_numpy_array(cls, xs: str) -> np.array:
        xl = list()
        s = 1
        for c in xs:
            if c == '-':
                s = -1
            else:
                xl.append(float(c) * float(s))
                s = 1
        return np.asarray(xl, dtype=np.float32)

    #
    # Convert Q Values from dict to numpy array, this is the training target Y
    #
    @classmethod
    def __qv_as_numpy_array(cls, qve: dict) -> np.array:
        sz = 9  # nine actions
        q_values = np.full(sz, np.float(0))
        for k, v in qve.items():
            q_values[int(k)] = np.float(v)
        return q_values

    #
    # Re Scale
    #
    @classmethod
    def __rescale(cls, v, mn, mx):
        return (v - mn) / (mx - mn)

    #
    # Load the States and Q Values as X,Y Training Set.
    #
    def load_state_qval_as_xy(self, filename: str) -> Tuple[np.array, np.array]:

        (qv, n, lr0, df, lrd) = self.__temporal_difference_policy_persistance.load(filename)

        x = np.zeros((len(qv), 9))  # 9 Board Cells
        y = np.zeros((len(qv), 9))  # 9 Q Vals, 1 for each action for a given board state.
        i = 0
        mn = np.finfo('d').max
        mx = -mn
        for qx, qy in qv.items():
            x[i] = self.state_as_str_to_numpy_array(qx)
            y[i] = self.__qv_as_numpy_array(qy)
            mx = np.max(y[i])
            mn = np.min(y[i])
            if mx - mn > 0:
                y[i] = (y[i] - mn) / (mx - mn)
            i += 1

        return x, y


# ********************
# *** UNIT TESTING ***
# ********************


class TestTDDeepNNPolicyPersistance(unittest.TestCase):

    def test_strategies(self):
        lg = EnvironmentLogging("TemporalDifferenceDeepNNPolicyPersistance",
                                "TemporalDifferenceDeepNNPolicyPersistance.log",
                                logging.DEBUG).get_logger()
        tdd = TemporalDifferenceDeepNNPolicyPersistance(lg=lg)
        x, y = tdd.load_state_qval_as_xy("./qvn_dump.pb")

        self.assertEqual(np.size(x), np.size(y))


#
# Execute the ReflrnUnitTests.
#


if __name__ == "__main__":
    tests = TestTDDeepNNPolicyPersistance()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
