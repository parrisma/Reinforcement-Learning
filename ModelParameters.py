
class ModelParameters:
    epochs = 100
    update_every_n_episodes = 100
    sample_size = 50  # episodes => about 500 events.
    batch_size = 10
    replay_mem_size = 1000
    save_every_n_critic_updates = 100

    def __init__(self,
                 epochs: int=100,
                 update_every_n_episodes: int=100,
                 sample_size: int=50,
                 batch_size: int=10,
                 replay_mem_size: int=1000,
                 save_every_n_critic_updates: int=100):

        self.epochs = epochs
        self.update_every_n_episodes = update_every_n_episodes
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.replay_mem_size = replay_mem_size
        self.save_every_n_critic_updates = save_every_n_critic_updates

    def get(self):
        return
