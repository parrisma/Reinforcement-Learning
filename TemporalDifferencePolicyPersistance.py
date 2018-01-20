import numpy as np

#
# Save and Load Q Value Dictionary of form
#   Key = State
#   Value = action, q-val pairs
#


class TemporalDifferencePolicyPersistance:

    def __init__(self):
        return

    #
    # Dump the given q values dictionary to a simple text dump.
    #
    @classmethod
    def save(cls, qv: dict, filename: str):
        out_f = None
        try:
            out_f = open(filename, "w")
            for state, q_val_dict in qv.items():
                out_f.write(state)
                out_f.write(":")
                for action, q_val in q_val_dict.items():
                    out_f.write(str(action)+':{:.16f}'.format(q_val) + "~")
                out_f.write("\n")
        except Exception as exc:
            print("Failed to save Q Values : " + str(exc))
            return False
        finally:
            out_f.close()
        return True
