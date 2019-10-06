import numpy as np
import torch
from src.rl_agents.model import DDQN_Model
from src.rl_agents.memory import Memory
from torch.nn.functional import mse_loss
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: Test / Add Cuda
device = torch.device('cuda')


class DQN:
    def __init__(self, state_dim, action_dim, gamma, hidd_ch, lr, eps, bs, target_interval, nproc, max_memory, her=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.eps = eps
        self.bs = bs
        self.target_interval = target_interval
        self.nproc = nproc
        self.memory = Memory(max_memory, nproc)
        self.policy = DDQN_Model(state_dim, action_dim, hidd_ch).to(device)
        self.target = DDQN_Model(state_dim, action_dim, hidd_ch).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)  # Only policy
        self.target.update_target(self.policy)
        self.target_count = 0
        self.her = her

    def act(self, obs, deterministic=False):
        x = torch.from_numpy(obs).float().to(device)
        action = self.policy.act(x, eps=self.eps if not deterministic else 0)
        return action

    def update(self):
        if len(self.memory.state) < self.bs // 4:
            return 0

        state, new_state, action, reward, is_terminal = self.memory.sample(self.bs, self.her)

        state = torch.tensor(state, dtype=torch.float).detach().to(device)
        new_state = torch.tensor(new_state, dtype=torch.float).detach().to(device)
        action = torch.tensor(action).detach().to(device)
        reward = torch.tensor(reward, dtype=torch.float).detach().to(device)
        is_terminal = torch.tensor(is_terminal, dtype=torch.float).detach().to(device)

        q = self.policy(state)[torch.arange(self.bs), action]
        max_action = torch.argmax(self.policy(new_state), dim=1)
        y = reward + self.gamma * self.target(new_state)[torch.arange(self.bs), max_action] * is_terminal
        loss = mse_loss(input=q, target=y.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_count += 1
        if self.target_count == self.target_interval:  # Sync the two
            self.target_count = 0
            self.target.update_target(self.policy)

        return loss

        
