import matplotlib.pyplot as plt


class Visualise:
    fig = None
    sub = None
    reward_func = 0
    qvals = 1
    probs = 2
    loss = 3
    plot_pause = 0.0001

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

        self.sub[self.reward_func].set_xlabel('X (State)')
        self.sub[self.reward_func].set_ylabel('Y (Reward)')
        self.sub[self.reward_func].set_title('Reward Function')

        self.sub[self.qvals].set_xlabel('X (State)')
        self.sub[self.qvals].set_ylabel('Y (Q-Values)')
        self.sub[self.qvals].set_title('Action Values')

        self.sub[self.probs].set_xlabel('X (State)')
        self.sub[self.probs].set_ylabel('Y (Action Probability)')
        self.sub[self.probs].set_title('Action Probabilities')

        self.sub[self.loss].set_xlabel('X (Training Episode)')
        self.sub[self.loss].set_ylabel('Y (Loss)')
        self.sub[self.loss].set_title('Training Loss')

        self.show()

        return

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
        self.sub[self.qvals].cla()
        self.sub[self.reward_func].plot(states, rewards)
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
        self.sub[self.qvals].cla()
        self.sub[self.qvals].scatter(states, qvalues_action1)
        self.sub[self.qvals].scatter(states, qvalues_action2)
        if qvalues_reference is not None:
            self.sub[self.qvals].scatter(states, qvalues_reference)
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
        self.sub[self.loss].cla()
        if actor_loss is not None:
            self.sub[self.loss].plot(list(range(0, len(actor_loss))), actor_loss)
        if critic_loss is not None:
            self.sub[self.loss].plot(list(range(0, len(critic_loss))), critic_loss)
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
        self.sub[self.probs].cla()
        self.sub[self.probs].scatter(states, action1_probs)
        self.sub[self.probs].scatter(states, action2_probs)
        self.show()
        return
