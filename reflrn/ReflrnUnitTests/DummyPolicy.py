from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


class DummyPolicy(Policy):
    def update_policy(self, agent_name: str, state: State, next_state: State, action: int, reward: float,
                      episode_complete: bool) -> None:
        pass

    def select_action(self, agent_name: str, state: State, possible_actions: [int]) -> int:
        pass

    def save(self, filename: str = None) -> None:
        pass

    def load(self, filename: str = None):
        pass
