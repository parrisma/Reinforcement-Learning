import unittest
from typing import Tuple

import numpy as np

from reflrn.Interface.TemporalDifferencePolicyPersistance import TemporalDifferencePolicyPersistance


#
# Save and Load Q Value Dictionary of form
#   Key = State
#   Value = action, q-val pairs
#


class TemporalDifferenceQValPolicyPersistance(TemporalDifferencePolicyPersistance):
    __enable_csv = False
    __max_actions_per_state = 4  # ToDo this needs to be passes in to save as CSV

    #
    # Enable / Disable csv file save.
    #
    @classmethod
    def enable_csv_file_save(cls) -> None:
        cls.__enable_csv = True

    @classmethod
    def disable_csv_file_save(cls) -> None:
        cls.__enable_csv = False

    #
    # Dump the given q values dictionary to a simple text dump.
    #
    @classmethod
    def save(cls,
             qv: dict,
             n: int,
             learning_rate_0: np.float,
             discount_factor: np.float,
             learning_rate_decay: np.float,
             filename: str) -> bool:

        cls.save_sparse(qv, n, learning_rate_0, discount_factor, learning_rate_decay, filename)
        if cls.__enable_csv:
            cls.save_as_csv(qv, cls.__with_csv_file_extension(filename))
        return True

    #
    # Dump the given q values dictionary to a simple text dump. This is a sparse format that is used to
    # restore the QV State.
    #
    @classmethod
    def save_sparse(cls,
                    qv: dict,
                    n: int,
                    learning_rate_0: np.float,
                    discount_factor: np.float,
                    learning_rate_decay: np.float,
                    filename: str) -> bool:
        out_f = None
        try:
            out_f = open(filename, "w")
            out_f.write(str(n) + "\n")
            out_f.write('{:.16f}'.format(learning_rate_0) + "\n")
            out_f.write('{:.16f}'.format(discount_factor) + "\n")
            out_f.write('{:.16f}'.format(learning_rate_decay) + "\n")
            for state, q_val_dict in qv.items():
                out_f.write(state)
                out_f.write("#")
                for action, q_val in q_val_dict.items():
                    out_f.write(str(action) + ':{:.16f}'.format(q_val) + "~")
                out_f.write("\n")
        except Exception as exc:
            print("Failed to save Q Values : " + str(exc))
            return False
        finally:
            out_f.close()
        return True

    #
    # Dump the given q values dictionary to a simple text dump, but full csv format.
    #
    @classmethod
    def save_as_csv(cls,
                    qv: dict,
                    filename: str) -> bool:
        out_f = None
        try:
            out_f = open(filename, "w")
            qvs = np.zeros(cls.__max_actions_per_state)
            for state, q_val_dict in qv.items():
                out_f.write(state)
                out_f.write(",")
                for action, q_val in q_val_dict.items():
                    qvs[action] = q_val
                for qv in qvs:
                    out_f.write('{:.16f}'.format(qv) + ",")
                out_f.write("\n")
                qvs = np.zeros(cls.__max_actions_per_state)
        except Exception as exc:
            print("Failed to save Q Values : " + str(exc))
            return False
        finally:
            out_f.close()
        return True

    #
    # Load the given file into a TD Policy state/action/q value dictionary. This loads the sparse
    # form of the save.
    #
    @classmethod
    def load(cls,
             filename: str) -> Tuple[dict, int, np.float, np.float, np.float]:
        n = learning_rate_0 = discount_factor = learning_rate_decay = np.float(0)
        in_f = None
        qv = dict()
        ln = 0
        try:
            in_f = open(filename, "r")
            with in_f as qv_dict_data:
                for line in qv_dict_data:
                    if ln == 0:
                        n = int(line)
                    elif ln == 1:
                        learning_rate_0 = np.float(line)
                    elif ln == 2:
                        discount_factor = np.float(line)
                    elif ln == 3:
                        learning_rate_decay = np.float(line)
                    else:
                        itms = line.split("#")
                        qv[itms[0]] = dict()
                        for qvs in itms[1][:-1].split("~"):
                            if len(qvs) > 0:
                                a, v = qvs.split(":")
                                qv[itms[0]][int(a)] = np.float(v)
                    ln += 1
        except Exception as exc:
            raise RuntimeError("Failed to load Q Values from file [" + filename + ": " + str(exc))
        finally:
            if in_f is not None:
                in_f.close()
        return qv, n, learning_rate_0, discount_factor, learning_rate_decay

    #
    # Change file extension to .csv
    #
    @classmethod
    def __with_csv_file_extension(cls, filename: str) -> str:
        bits = filename.split('.')
        ln = len(bits)

        csv_filename = ''
        if ln >= 1:
            for i in range(0, ln - 1):
                csv_filename += bits[i] + '.'
            csv_filename += 'csv'
        else:
            csv_filename = filename + '.csv'
        return csv_filename


# ********************
# *** UNIT TESTING ***
# ********************


class TestTemporalDifferencePolicyPersistance(unittest.TestCase):

    def test_load_and_save(self):
        tdpp = TemporalDifferenceQValPolicyPersistance()
        ld = tdpp.load("./qvn_dump.pb")
        tdpp.save_as_csv(ld, "./qvn_dump.csv")


#
# Execute the ReflrnUnitTests.
#


if __name__ == "__main__":
    tests = TestTemporalDifferencePolicyPersistance()
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    unittest.TextTestRunner().run(suite)
