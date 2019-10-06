import numpy as np

class Memory:
    def __init__(self, max_memory, max_proc):
        self.max_memory = max_memory
        self.max_proc = max_proc
        self.state = []
        self.new_state = []
        self.action = []
        self.reward = []
        self.is_terminal = []
        self._tempbuffer = [[] for _ in range(self.max_proc)]

    def store_transition(self, s, s1, a, r, is_terminal, store):
        for i in range(len(s)):
            if store[i]:
                self._tempbuffer[i].append([s[i], s1[i], a[i], r[i], is_terminal[i]])
                if is_terminal[i]:
                    ep = np.array(self._tempbuffer[i]).T
                    self._store_episode(ep[0], ep[1], ep[2], ep[3], ep[4])
                    del self._tempbuffer[i][:]

    def _store_episode(self, s, s1, a, r, is_terminal):
        self.state.append(s)
        self.new_state.append(s1)
        self.action.append(a)
        self.reward.append(r)
        self.is_terminal.append(is_terminal)
        assert len(self.state) == len(self.new_state) == len(self.reward) == len(self.is_terminal) == len(self.action)

        if len(self.state) > self.max_memory:
            self.state.pop(0)
            self.new_state.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.is_terminal.pop(0)

    def clear_memory(self):
        del self.state[:]
        del self.new_state[:]
        del self.action[:]
        del self.reward[:]
        del self.is_terminal[:]
        del self.indexer[:]

    def sample(self, bs, her):
        idx = np.random.randint(len(self.state), size=bs)
        if her is not None:
            her_idx = idx[bs - her.replay_k:]
            idx = idx[:bs - her.replay_k]
            sample_idx = [np.random.randint(len(self.state[i])) for i in idx]

        state, new_state, action, reward, is_terminal = [], [], [], [], []
        for i, s in zip(idx, sample_idx):
            state.append(self.state[i][s])
            new_state.append(self.new_state[i][s])
            action.append(self.action[i][s])
            reward.append(self.reward[i][s])
            is_terminal.append(1 - int(self.is_terminal[i][s]))

        if her is not None:
            r_state, r_new_state, r_action, r_reward, r_is_terminal = her.sample(self, her_idx)
            state = np.concatenate([state, r_state], axis=0)
            new_state = np.concatenate([new_state, r_new_state], axis=0)
            action = np.concatenate([action, r_action], axis=0)
            reward = np.concatenate([reward, r_reward], axis=0)
            is_terminal = np.concatenate([is_terminal, r_is_terminal], axis=0)
        return state, new_state, action, reward, is_terminal
