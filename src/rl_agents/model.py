import torch
from torch import nn
import numpy as np


class DDQN_Model(nn.Module):
    def __init__(self, state_size, action_size, hidd_ch=128):
        super(DDQN_Model, self).__init__()
        self.action_size = action_size

        self.features = nn.Sequential(
            nn.Linear(state_size, hidd_ch),
            nn.ReLU(),
            #nn.Linear(hidd_ch, hidd_ch),
            #nn.ReLU()
        )

        self.advantage = nn.Sequential(
            #nn.Linear(hidd_ch, hidd_ch),
            #nn.ReLU(),
            nn.Linear(hidd_ch, self.action_size)
        )

        self.value = nn.Sequential(
            #nn.Linear(hidd_ch, hidd_ch),
            #nn.ReLU(),
            nn.Linear(hidd_ch, 1)
        )

    def forward(self, obs):
        x = self.features(obs.float())
        adv = self.advantage(x)
        value = self.value(x)
        return value + (adv - adv.mean(-1, keepdim=True))

    def act(self, state, eps):
        if np.random.random() > eps:
            q = self.forward(state)
            action = torch.argmax(q, dim=-1).cpu().data.numpy()
        else:
            action = np.random.randint(self.action_size, size=1 if len(state.shape) == 1 else state.shape[0])
        return action

    def update_target(self, model):
        self.load_state_dict(model.state_dict())
