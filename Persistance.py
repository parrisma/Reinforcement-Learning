import numpy as np

#
# Save and Load Q Value Dictionary of form
#   Key = String
#   Value = numpy array(9,float)
#


class Persistance:

    def __init__(self):
        return

    #
    # Dump the given q values dictionary to a simple text dump.
    #
    @classmethod
    def save(cls, qv, filename):
        out_f = None
        try:
            out_f = open(filename, "w")
            for state, qvals in qv.items():
                out_f.write(state)
                out_f.write(":")
                for i in range(0, len(qvals)):
                    out_f.write('{:.16f}'.format(qvals[i]) + ":")
                out_f.write("\n")
        except Exception as exc:
            print("Failed to save Q Values : " + str(exc))
            return False
        finally:
            out_f.close()
        return True

    #
    # Load the given text dump of the q values
    #
    @classmethod
    def load(cls, filename):
        in_f = None
        qv = dict()
        try:
            s_nan = str(np.nan)
            in_f = open(filename, "r")
            with in_f as qv_dict_data:
                for line in qv_dict_data:
                    itms = line.split(":")
                    qvs = np.full(9, np.nan)
                    i = 0
                    for fpn in itms[1:10]:
                        if fpn != s_nan:
                            qvs[i] = float(fpn)
                        i += 1
                    qv[itms[0]] = qvs
        except Exception as exc:
            print("Failed to load Q Values : " + str(exc))
            return None
        finally:
            in_f.close()
        return qv

    @classmethod
    def x_as_str_to_num_array(cls,xs):
        xl = list()
        s = 1
        for c in xs:
            if c == '-':
                s = -1
            else:
                xl.append(float(c)*float(s))
                s = 1
        return np.asarray(xl, dtype=np.float32)

    #
    # Load as an X Y Network training set. Both are 9 by 1
    #
    @classmethod
    def load_as_X_Y(cls, filename):
        qv = cls.load(filename)
        x = np.zeros((len(qv, 9)))
        y = np.zeros((len(qv, 9)))
        i = 0
        mn = np.finfo('d').max
        mx = -mx
        for qx, qy in qv:
            x[i] = cls.x_as_str_to_num_array(qx)
            y[i] = qy
            mx = np.max(mx, np.max(qv))
            mn = np.max(mn, np.max(qv))
            i += 1

            mn *= 1.1

        for qy in y:
            qy[np.isnan(qy) == True] = mn
            qy *=

