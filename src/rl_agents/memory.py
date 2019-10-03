import numpy as np

class Memory:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.state = []
        self.new_state = []
        self.action = []
        self.reward = []
        self.is_terminal = []
        self.indexer = [0]

    def store_transition(self, s, s1, a, r, is_terminal):
        self.state.append(s)
        self.new_state.append(s1)
        self.action.append(a)
        self.reward.append(r)
        self.is_terminal.append(is_terminal)
        self.indexer.append(int(not is_terminal))
        assert len(self.state) == len(self.new_state) == len(self.reward) == len(self.is_terminal) == len(self.action) == len(self.indexer) - 1

        if len(self.state) > self.max_memory:
            self.state.pop(0)
            self.new_state.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.is_terminal.pop(0)
            self.indexer.pop(1)

    def clear_memory(self):
        del self.state[:]
        del self.new_state[:]
        del self.action[:]
        del self.reward[:]
        del self.is_terminal[:]
        del self.indexer[:]

    def assign_idx(self, sampled_idx):
        idx = [i for i in range(len(self.indexer)) if self.indexer[i] == 0]
        if self.indexer[-1] != 0:
            idx.append(len(self.indexer) - 1)
        idx = [[[idx[i - 1], idx[i]] for i in range(len(idx)) if idx[max(0, i - 1)] <= s and idx[i] > s] for s in sampled_idx]
        return np.array(idx).reshape((len(sampled_idx), 2))

    def sample(self, bs, her):
        idx = np.random.randint(len(self.state), size=bs)
        if her is not None:
            her_idx = idx[bs - her.replay_k:]
            rel_idx = self.assign_idx(her_idx)
            idx = idx[:bs - her.replay_k]

        state = [self.state[i] for i in idx]
        new_state = [self.new_state[i] for i in idx]
        action = [self.action[i] for i in idx]
        reward = [self.reward[i] for i in idx]
        is_terminal = [1 - int(self.is_terminal[i]) for i in idx]
        if her is not None:
            r_state, r_new_state, r_action, r_reward, r_is_terminal = her.sample(self, her_idx, rel_idx)
            state = np.concatenate([state, r_state], axis=0)
            new_state = np.concatenate([new_state, r_new_state], axis=0)
            action = np.concatenate([action, r_action], axis=0)
            reward = np.concatenate([reward, r_reward], axis=0)
            is_terminal = np.concatenate([is_terminal, r_is_terminal], axis=0)
        return state, new_state, action, reward, is_terminal
