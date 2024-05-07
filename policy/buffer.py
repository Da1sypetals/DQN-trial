from collections import deque
import random
import torch
from typing import List
import numpy as np



class Transition:
    def __init__(self,
                 state,
                 action,
                 reward,
                 next_state,
                 done):
        
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class BatchedTransition:
    def __init__(self, transitions: List[Transition]):
        self.rewards = torch.tensor(np.array([t.reward for t in transitions]), dtype=torch.float32)
        self.actions = torch.tensor(np.array([t.action for t in transitions]), dtype=int)
        self.states = torch.tensor(np.array([t.state for t in transitions]), dtype=torch.float32)
        self.next_states = torch.tensor(np.array([t.next_state for t in transitions]), dtype=torch.float32)
        self.dones = torch.tensor(np.array([t.done for t in transitions]), dtype=torch.float32)


    def to(self, device):
        self.rewards = self.rewards.to(device)
        self.actions = self.actions.to(device)
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.dones = self.dones.to(device)




class Buffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, element):
        self.buffer.append(element)

    def draw(self, n) -> BatchedTransition:
        if n <= 0:
            raise ValueError("n must be positive")
        if n > self.capacity:
            raise ValueError("n must be less than or equal to the Buffer's capacity")
        
        n_samples = min(n, len(self.buffer))
        transition_list = random.sample(self.buffer, n_samples)

        return BatchedTransition(transition_list)
    
    def __len__(self):
        return len(self.buffer)


