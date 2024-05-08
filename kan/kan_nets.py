from .impl import KAN, KANLayer
import torch
import torch.nn as nn



def make_kan_nets(obs_dim, action_dim, device):
    q_net = Net(obs_dim, action_dim).to(device)
    target_net = Net(obs_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net = target_net.to(device)
    return q_net, target_net


class Net(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.action_dim = action_dim

        self.base = KAN([obs_dim, 256, 256, 64])

        self.action_advantage_head = KAN([64, 64, action_dim])
        self.state_value_head = KAN([64, 64, 1])



    def forward(self, x):
        x = self.base(x)

        # distribution of actions
        advantage = self.action_advantage_head(x)

        state_value = self.state_value_head(x)

        return advantage, state_value








