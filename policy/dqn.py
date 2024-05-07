from copy import deepcopy
import torch
import torch.nn as nn
from .buffer import Buffer, BatchedTransition
import random
from .net import Net
import numpy as np
from typing import List
import einops as ein



class DQN:

    def __init__(self,
                 q_net: Net,
                 target_net: Net, 
                 q_optim: torch.optim.Optimizer,
                 gamma=.98,
                 epsilon=.02,
                 device=torch.device('cuda')):
        
        self.q_net = q_net
        self.target_net = target_net

        self.epsilon = epsilon
        self.gamma = gamma

        self.device = device

        self.q_optim = q_optim


    def take_action(self, state):
        
        if random.random() < self.epsilon:
            action = random.randint(0, self.q_net.action_dim - 1)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            advantage, state_value = self.q_net(state)
            action = torch.argmax(advantage, dim=-1)

        return torch.tensor(action)
    

    def update_transitions(self, td: BatchedTransition):

        self.q_optim.zero_grad()

        target_advantage, target_state_value = self.target_net(td.next_states)
        max_target_advantage, _ = torch.max(target_advantage, dim=-1)
        max_target_q = target_state_value.view(-1) + max_target_advantage - torch.mean(target_advantage, dim=-1)

        # print(max_target_q.size())

        advantage, state_value = self.q_net(td.states)
        q = state_value.view(-1) + \
                        (advantage.gather(1, ein.rearrange(td.actions, 'n -> n 1')).view(-1) - 
                        torch.mean(advantage, dim=-1)).view(-1)

        # print(q.mean())

        td_target = td.rewards.view(-1) + self.gamma * max_target_q * (1 - td.dones)

        q_loss = nn.functional.mse_loss(td_target, q, reduction='mean')
        q_loss.backward()
        self.q_optim.step()



    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())












