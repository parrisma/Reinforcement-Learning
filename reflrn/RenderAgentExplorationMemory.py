from reflrn.Interface.ExplorationMemory import ExplorationMemory
from reflrn.Interface.RenderExplorationMemories import RenderExplorationMemory


class RenderAgentExplorationMemory(RenderExplorationMemory):
    def render_episode(self, exploration_memory: ExplorationMemory, episode: int) -> str:

        memory = exploration_memory.get_memories_by_episode(episode=episode)
        ep_len = 0
        ep_cost = float(0)
        s = ''
        if memory is not None:
            ep_len = len(memory)
            for mem in memory:
                ep_cost += mem[ExplorationMemory.Memory.REWARD]
            s = 'Episode Summary : Length: ' + str(ep_len) + ' Cost : ' + str(ep_cost)

        return s
