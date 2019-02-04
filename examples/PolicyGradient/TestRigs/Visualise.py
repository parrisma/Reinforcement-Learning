import matplotlib.pyplot as plt


class Visualise:
    fig = None
    sub = None
    reward_func = None
    qvals = None
    qvals_x2 = None
    probs = None
    loss = None
    loss_x2 = None
    plot_pause = 0.0001
    nxt_col = 0
    plt_col = dict()

    def __init__(self):
        """
        Create a set of 4 sub-plots in a column
            Reward Function
            Q Values
            Probabilities
            Training Loss(es)
        """
        self.fig, self.sub = plt.subplots(4)
        self.fig.suptitle('Actor Critic Telemetry')

        self.reward_func = self.sub[0]
        self.reward_func.set_xlabel('X (State)')
        self.reward_func.set_ylabel('Y (Reward)')
        self.reward_func.set_title('Reward Function')

        self.qvals = self.sub[1]
        self.qvals.set_xlabel('X (State)')
        self.qvals.set_ylabel('Y (Q-Values Predicted)')
        self.qvals.set_title('Action Values')
        self.qvals_x2 = self.qvals.twinx()
        self.qvals_x2.set_ylabel('Y (Q-Values Ref Func)')

        self.probs = self.sub[2]
        self.probs.set_xlabel('X (State)')
        self.probs.set_ylabel('Y (Action Probability)')
        self.probs.set_title('Action Probabilities')

        self.loss = self.sub[3]
        self.loss.set_xlabel('X (Training Episode)')
        self.loss.set_ylabel('Y (Actor Loss)')
        self.loss.set_title('Training Loss')
        self.loss_x2 = self.loss.twinx()
        self.loss_x2.set_ylabel('Y (Critic Loss)')
        self.show()

        return

    def plt_color(self,
                  plt_id: int) -> str:
        """
        :param: pt_id a unique numerical id for the plot with which to associate a color
        A random color code to use for setting line colours for the given plot id
        :return: color in form 'C<n>' where n is a integer that increased by 1 for every call.
        """
        if plt_id not in self.plt_col:
            self.plt_col[plt_id] = "C{:d}".format(self.nxt_col)
            self.nxt_col += 1
        return self.plt_col[plt_id]

    def show(self) -> None:
        """
        Render the Figure + Sub Plots
        :return: None
        """
        plt.pause(self.plot_pause)
        plt.show(block=False)
        return

    def plot_reward_function(self,
                             states: list,
                             rewards: list) -> None:
        """
        Render or Update the Sub Plot for Reward Function
        :param states: Reward Function States
        :param rewards: Rewards for given States
        :return: Nothing
        """
        self.reward_func.cla()
        self.reward_func.plot(states, rewards, color=self.plt_color(1))
        self.show()
        return

    def plot_qvals_function(self,
                            states: list,
                            qvalues_action1: list,
                            qvalues_action2: list,
                            qvalues_reference: list = None
                            ) -> None:
        """
        Render or Update the Sub Plot for learned Q-values
        :param states: Q Values States
        :param qvalues_action1: Q Values for action 1 from given States
        :param qvalues_action2: Q Values for action2 from given States
        :param qvalues_reference: Q Value (actual) from given state
        :return: Nothing
        """
        self.qvals.cla()
        self.qvals_x2.cla()
        self.qvals.plot(states, qvalues_action1, color=self.plt_color(2))
        self.qvals.plot(states, qvalues_action2, color=self.plt_color(3))
        if qvalues_reference is not None:
            self.qvals_x2.plot(states, qvalues_reference, color=self.plt_color(4))
        self.show()
        return

    def plot_loss_function(self,
                           actor_loss: list = None,
                           critic_loss: list = None
                           ) -> None:
        """
        Render or Update the sub plot for actor / critic loss
        :param actor_loss: Actor Loss Time Series
        :param critic_loss: Critic Loss Time Series
        :return: Nothing
        """
        if actor_loss is not None:
            self.loss.cla()
            self.loss.plot(list(range(0, len(actor_loss))), actor_loss, color=self.plt_color(5))
        if critic_loss is not None:
            self.loss_x2.cla()
            self.loss_x2.plot(list(range(0, len(critic_loss))), critic_loss, color=self.plt_color(6))
        self.show()
        return

    def plot_prob_function(self,
                           states: list,
                           action1_probs: list,
                           action2_probs: list
                           ) -> None:
        """
        Render or Update the sub plot for action 1 & 2 probability Distributions by state
        :param states
        :param action1_probs probabilities by state
        :param action2_probs probabilities by state
        :return: Nothing
        """
        self.probs.cla()
        self.probs.plot(states, action1_probs, color=self.plt_color(7))
        self.probs.plot(states, action2_probs, color=self.plt_color(8))
        self.show()
        return
