import numpy as np

class HER:
    def __init__(self, replay_strategy, goal_size, replay_k):
        self.replay_strategy = replay_strategy
        self.goal_size = goal_size
        assert self.replay_strategy in ['future', 'episode', 'random', 'final']
        self.replay_k = replay_k

    def sample(self, memory, idx):
        state_size = len(memory.state[0][0])
        ep_length = [len(memory.state[i]) for i in idx]
        samples = [np.random.randint(ep) for ep in ep_length]
        state, new_state, action, reward, is_terminal, new_goal = [], [], [], [], [], []
        for i, ep, picked in zip(range(len(idx)), idx, samples):
            state.append(memory.state[ep][picked][:state_size - self.goal_size])
            new_state.append(memory.new_state[ep][picked][:state_size - self.goal_size])
            action.append(memory.action[ep][picked])
            is_terminal.append(1 - int(memory.is_terminal[ep][picked]))

            if self.replay_strategy == 'final':
                new_goal.append(memory.new_state[ep][-1][:state_size - self.goal_size])
            elif self.replay_strategy == 'episode':
                p_idx = np.random.randint(ep_length[i])
                new_goal.append(memory.new_state[ep][p_idx][:state_size - self.goal_size])
            elif self.replay_strategy == 'future':
                p_idx = np.random.randint(samples[i], ep_length[i])
                new_goal.append(memory.new_state[ep][p_idx][:state_size - self.goal_size])
            elif self.replay_strategy == 'random':
                ep_random = np.random.randint(len(memory.state))
                p_idx = np.random.randint(len(memory.state[ep_random]))
                new_goal.append(memory.new_state[ep_random][p_idx][:state_size - self.goal_size])

            reward.append(1 if all(new_state[-1] == new_goal[-1]) else 0)

        state = np.concatenate([state, new_goal], axis=1)
        new_state = np.concatenate([new_state, new_goal], axis=1)

        return state, new_state, action, reward, is_terminal

