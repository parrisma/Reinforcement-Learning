import numpy as np

from reflrn.Interface.RenderQVals import RenderQVals
from reflrn.Interface.State import State


class RenderQValues(RenderQVals):

    #
    # Normalise
    #
    @classmethod
    def __normalise(cls,
                    a: np.array
                    ) -> np.array:
        qn = np.copy(a)
        qn = np.where(np.isnan(qn), 0, qn)
        qn[~np.isinf(qn)] -= np.min(qn[~np.isinf(qn)], axis=0)
        qn[~np.isinf(qn)] /= np.ptp(qn[~np.isinf(qn)], axis=0)
        qn[~np.isinf(qn)] *= float(100)
        return qn

    #
    # render qval array as string
    #
    @classmethod
    def render_qval_array(cls,
                          q_vals: np.array
                          ) -> str:
        at = 0
        s = str()
        q = np.reshape(np.copy(q_vals), 9)
        qn = RenderQValues.__normalise(q)
        for i in range(0, 3):
            for j in range(0, 3):
                v = qn[at]
                if np.isinf(v):
                    s += "[(" + "---" + "%) " + '{:+.8}'.format(qn[at]) + "] "
                else:
                    s += "[(" + '{:+3d}'.format(int(cls.__nan_2_n(v, float(0)))) + "%) " + \
                         '{:+.8}'.format(cls.__nan_2_n(q[at], float(0))) + "] "
                at += 1
            s += "\n"
        return s

    #
    # Render the given Q Values for the given state as a string.
    #
    def render(self,
               curr_state: State,
               q_vals: dict) -> str:
        s = ""
        at = 0

        q, a = self.__get_q_vals_as_np_array(curr_state)

        if a is not None:
            a = np.sort(a)
        mxq = np.max(q)
        for i in range(0, 3):
            for j in range(0, 3):
                if a is not None and at < len(a) and a[at] == j + (i * 3):
                    qpct = (q[at] / mxq) * 100
                    if np.isnan(qpct):
                        qpct = 0
                    s += "[(" + '{:+3d}'.format(int(qpct)) + "%) " + '{:+.16f}'.format(q[at]) + "] "
                    at += 1
                else:
                    s += "[                           ] "
            s += "\n"
        return s

    #
    # get_memories_by_type q values and associated actions as numpy array
    #
    @classmethod
    def __get_q_vals_as_np_array(cls, q_val_dict: dict, state: State) -> np.array:
        q_values = None
        q_actions = None

        # If there are no Q values learned yet we cannot predict a greedy action.
        if q_val_dict is not None:
            state_name = state.state_as_string()

            if state_name in q_val_dict:
                sz = len(q_val_dict[state_name])
                q_values = np.full(sz, np.nan)
                q_actions = np.array(sorted(list(q_val_dict[state_name].keys())))
                i = 0
                for actn in q_actions:
                    q_values[i] = q_val_dict[state_name][actn]
                    i += 1

        return q_values, q_actions

    #
    # replace NaN with a valid number
    #
    @classmethod
    def __nan_2_n(cls,
                  v,
                  n):
        if np.isnan(v):
            return n
        return v
