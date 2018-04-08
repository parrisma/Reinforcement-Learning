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
    # Re Scale
    #
    @classmethod
    def rescale(cls, v, mn, mx):
        return (v-mn)/(mx-mn)

    #
    # Load as an X Y Network training set. Both are 9 by 1
    #
    # X (board state + player) is converted to array of (1,0,-1)'s
    # Y (Q Vals) is returned as is but with nan set to zero & re-scaled w.r.t entire Q Val array
    #
    @classmethod
    def load_as_X_Y(cls, filename):
        qv = cls.load(filename)
        x = np.zeros((len(qv), 1+9))  # Player + 9 Board Cells
        y = np.zeros((len(qv), 9)) # 9 Q Vals.
        i = 0
        mn = np.finfo('d').max
        mx = -mn
        for qx, qy in qv.items():
            x[i] = cls.x_as_str_to_num_array(qx)
            y[i] = qy
            mx = max(mx, np.max(qy[np.isnan(qy)==False]))
            mn = min(mn, np.min(qy[np.isnan(qy)==False]))
            i += 1

        mn *= 1.1
        i = 0
        for qy in y:
            qy[np.isnan(qy) == True] = mn
            y[i] = cls.rescale(qy, mn, mx)
            i += 1

        return x, y
