import numpy as np
from typing import Tuple

#
# Save and Load Q Value Dictionary of form
#   Key = State
#   Value = action, q-val pairs
#


class TemporalDifferencePolicyPersistance:

    def __init__(self):
        return

    __n = 0  # number of learning events
    __learning_rate_0 = float(0.05)
    __discount_factor = float(0.8)
    __learning_rate_decay = float(0.001)

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
                    out_f.write(str(action)+':{:.16f}'.format(q_val) + "~")
                out_f.write("\n")
        except Exception as exc:
            print("Failed to save Q Values : " + str(exc))
            return False
        finally:
            out_f.close()
        return True

    #
    # Load the given file into a TD Policy state/action/q value dictionary
    #
    @classmethod
    def load(cls, filename: str) -> Tuple[dict, int, np.float, np.float, np.float]:
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
