import numpy as np

class HER:
    def __init__(self, replay_strategy, goal_size, replay_k):
        self.replay_strategy = replay_strategy
        self.goal_size = goal_size
        assert self.replay_strategy in ['future', 'episode', 'random', 'final']
        self.replay_k = replay_k

    def sample(self, memory, idx, rel_idx):
        state_size = len(memory.state[0])
        state = np.array([memory.state[i] for i in idx])[..., :state_size - self.goal_size]
        new_state = np.array([memory.new_state[i] for i in idx])[..., :state_size - self.goal_size]
        action = [memory.action[i] for i in idx]
        is_terminal = [1 - int(memory.is_terminal[i]) for i in idx]

        if self.replay_strategy == 'final':
            new_goal = [memory.new_state[end - 1][:state_size - self.goal_size] for start, end in rel_idx]
        elif self.replay_strategy == 'episode':
            picked_idx = [np.random.randint(start, end) for start, end in rel_idx]
            new_goal = [memory.new_state[i][:state_size - self.goal_size] for i in picked_idx]
        elif self.replay_strategy == 'future':
            if idx in rel_idx:
                picked_idx = idx - 1
            else:
                picked_idx = [np.random.randint(start, end) for start, end in zip(idx, rel_idx[..., 1])]
            new_goal = [memory.new_state[i][:state_size - self.goal_size] for i in picked_idx]
        elif self.replay_strategy == 'random':
            picked_idx = [np.random.randint(0, len(memory.state)) for _ in range(len(idx))]
            new_goal = [memory.new_state[i][:state_size - self.goal_size] for i in picked_idx]

        reward = [1 if all(new_state[i] == new_goal[i]) else 0 for i in range(len(idx))]
        state = np.concatenate([state, new_goal], axis=1)
        new_state = np.concatenate([new_state, new_goal], axis=1)

        return state, new_state, action, reward, is_terminal

