import torch
import torch.nn as nn
from copy import deepcopy


def make_nets(obs_dim, action_dim, device):
    q_net = Net(obs_dim, action_dim).to(device)
    target_net = Net(obs_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net = target_net.to(device)
    return q_net, target_net


class Net(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.action_dim = action_dim

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)

        self.action_advantage_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.state_value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)

        # distribution of actions
        advantage = self.action_advantage_head(x)

        state_value = self.state_value_head(x)

        return advantage, state_value









