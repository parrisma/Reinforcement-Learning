import logging
from collections import deque

from reflrn.Interface.ExplorationMemory import ExplorationMemory
from reflrn.Interface.Policy import Policy
from reflrn.Interface.State import State


#
# Store memories in a DEQUE and index such that memories can be extracted by episode id or Memory Type
#

class AgentExplorationMemory(ExplorationMemory):
    __START = 0
    __END = 1

    #
    # Initialise the Deque to requested size.
    #
    def __init__(self,
                 lg: logging,
                 replay_mem_size: int = 100000):  # 100K Entries Max
        self.__memory = deque([], maxlen=replay_mem_size)
        self.__lg = lg
        self.__index = dict()
        self.__index[ExplorationMemory.Memory.EPISODE] = dict()
        self.__index[ExplorationMemory.Memory.POLICY] = dict()
        self.__index[ExplorationMemory.Memory.AGENT] = dict()
        self.__index[ExplorationMemory.Memory.STATE] = dict()
        self.__index[ExplorationMemory.Memory.ACTION] = dict()
        self.__current_episode_id = None
        return

    #
    # Add to the DEQUE and index.
    #
    def add(self,
            episode_id: int,
            policy: Policy,
            agent_name: str,
            state: State,
            next_state: State,
            action: int,
            reward: float,
            episode_complete: bool):

        # Added in the order of the offsets as defined in
        # ExplorationMemory.Memory
        self.__memory.append((episode_id,
                              policy,
                              agent_name,
                              state,
                              next_state,
                              action,
                              reward,
                              episode_complete)
                             )
        # A subset of the overall fields are indexed as per the ExplorationMemory.SUPPORTED_GETBY_INDEX list.
        # in addition was always add the episode as an index.
        self.__track_episodes(episode_id)
        self.__update_index(ExplorationMemory.Memory.POLICY, policy, episode_id)
        self.__update_index(ExplorationMemory.Memory.AGENT, agent_name, episode_id)
        self.__update_index(ExplorationMemory.Memory.STATE, state, episode_id)
        self.__update_index(ExplorationMemory.Memory.ACTION, action, episode_id)

        return

    #
    # Get the set of memories for the given memory type that can be found in the
    # given episode.
    #
    def get_memories_by_type(self,
                             get_by: ExplorationMemory.Memory,
                             value: object,
                             last_only: bool = False
                             ) -> [[], [], [], [], [], [], [], []]:

        if get_by not in ExplorationMemory.SUPPORTED_GETBY_INDEX:
            raise ExplorationMemory.ExplorationMemoryMemTypeSearchNotSupported(str(get_by) + " Not Searchable")

        return self.__get_memories(self.__get_episodes_for_memory(get_by,
                                                                  value,
                                                                  last_only
                                                                  ),
                                   get_by,
                                   value
                                   )

    #
    # Get the set of memories that correspond to the given episode.
    #
    def get_memories_by_episode(self, episode: int) -> [[], [], [], [], [], [], [], []]:
        return self.__get_memories([episode])

    #
    # Keep track of the start and end of every episode in the Deque
    #
    def __track_episodes(self,
                         episode_id: int) -> None:

        if self.__current_episode_id != episode_id:
            if self.__current_episode_id is not None:
                self.__episode_end(self.__current_episode_id, len(self.__memory) - 1)
            self.__episode_start(episode_id, len(self.__memory) - 1)
        return

    #
    # Record the start position of a new episode
    #
    def __episode_start(self,
                        episode_id: int,
                        start_idx: int) -> None:

        if episode_id not in self.__index[ExplorationMemory.Memory.EPISODE]:
            self.__index[ExplorationMemory.Memory.EPISODE][episode_id] = [start_idx, -1]
            self.__current_episode_id = episode_id
        return

    #
    # Record the start position of a new episode
    #
    def __episode_end(self,
                      episode_id: int,
                      end_idx: int) -> None:

        eidx = self.__index[ExplorationMemory.Memory.EPISODE][episode_id]
        eidx[self.__END] = end_idx
        self.__index[ExplorationMemory.Memory.EPISODE][episode_id] = eidx

        return

    #
    # Keep a track of all the episodes the given key appears in.
    #
    def __update_index(self,
                       idx_type: ExplorationMemory.Memory,
                       idx_key,
                       episode_id: int) -> None:

        if idx_key not in self.__index[idx_type]:
            self.__index[idx_type][idx_key] = list()
        if episode_id not in self.__index[idx_type][idx_key]:
            self.__index[idx_type][idx_key].append(episode_id)
        return

    #
    # return the list of episodes for the given memory type, value
    #
    def __get_episodes_for_memory(self,
                                  get_by: ExplorationMemory.Memory,
                                  value: object,
                                  last_only: bool = False) -> []:

        if value is None:
            raise ValueError('Value cannot be None')

        episodes = None
        if get_by in self.__index:
            if value in self.__index[get_by]:
                episodes = self.__index[get_by][value]

        if episodes is not None and last_only:
            episodes = [episodes[-1]]

        return episodes

    #
    # Return all the Deque entries for the given list of episode id's
    #
    def __get_memories(self,
                       episodes: [],
                       get_by: ExplorationMemory.Memory = None,
                       value: object = None) -> [[], [], [], [], [], [], [], []]:

        if episodes is None:
            return None

        episode_deque = deque([])
        sorted_episodes = sorted(episodes)
        for episode in sorted_episodes:
            if episode not in self.__index[ExplorationMemory.Memory.EPISODE]:
                raise ExplorationMemory.ExplorationMemoryNoSuchEpisode(str(episode) + " does not exist in memory")
            st_idx, ed_idx = self.__index[ExplorationMemory.Memory.EPISODE][episode]
            if ed_idx == -1:
                ed_idx = len(self.__memory)
            for idx in range(st_idx, ed_idx):
                if get_by is not None and value is not None:
                    if self.__memory_match(self.__memory[idx], get_by, value):
                        episode_deque.append(self.__memory[idx])
                else:
                    episode_deque.append(self.__memory[idx])

        cols = [[None for i in range(8)] for j in
                range(len(episode_deque))]  # episode_id, policy, agent, state, next_state,
        # action,
        # reward
        i = 0
        j = 0
        for mem in episode_deque:
            for j in range(0, len(mem)):
                cols[i][j] = mem[j]
            i += 1

        return cols

    #
    # Does the given memory match by just the given memory type and value.
    #
    @classmethod
    def __memory_match(cls, memory: [[], [], [], [], [], [], [], []],
                       get_by: ExplorationMemory.Memory,
                       value: object) -> bool:
        return memory[get_by] == value
