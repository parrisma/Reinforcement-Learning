import numpy as np

from reflrn.RenderQValsAsStr import RenderQValsAsStr
from reflrn.State import State


class RenderQValuesAsStr(RenderQValsAsStr):

    def render_as_str(self, curr_state: State, q_vals: dict) -> str:
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
    # get q values and associated actions as numpy array
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
